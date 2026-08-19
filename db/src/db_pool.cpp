#include "db_pool.h"
#include <spdlog/spdlog.h>
#include <sstream>
#include <algorithm>

namespace hms {

DbPool::DbPool(const Config& config) : pool_size_(config.pool_size) {
    std::ostringstream oss;
    oss << "host=" << config.host
        << " port=" << config.port
        << " user=" << config.user
        << " password=" << config.password
        << " dbname=" << config.database;
    conn_string_ = oss.str();

    spdlog::info("Initializing database pool (size={}) to {}:{}/{}",
                 pool_size_, config.host, config.port, config.database);

    // Pre-create connections
    for (int i = 0; i < pool_size_; ++i) {
        try {
            pool_.push(create_connection());
            ++total_created_;
        } catch (const std::exception& e) {
            spdlog::error("Failed to create DB connection {}/{}: {}", i + 1, pool_size_, e.what());
            // Continue - pool can work with fewer connections. acquire() will
            // opportunistically backfill the missing slots later (with
            // backoff) once the database becomes reachable.
        }
    }

    spdlog::info("Database pool initialized with {}/{} connections", total_created_, pool_size_);
}

DbPool::~DbPool() {
    std::lock_guard lock(mutex_);
    while (!pool_.empty()) {
        pool_.pop();
    }
}

std::unique_ptr<pqxx::connection> DbPool::create_connection() {
    auto conn = std::make_unique<pqxx::connection>(conn_string_);
    if (!conn->is_open()) {
        throw std::runtime_error("Failed to open database connection");
    }
    return conn;
}

DbPool::ConnectionGuard DbPool::acquire() {
    std::unique_ptr<pqxx::connection> conn;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);

    std::unique_lock lock(mutex_);
    while (true) {
        if (!pool_.empty()) {
            conn = std::move(pool_.front());
            pool_.pop();
            break;
        }

        auto now = std::chrono::steady_clock::now();

        // The pool is empty. If it never reached pool_size_ (e.g. the database
        // was unreachable when some initial connections were created, or a
        // connection was dropped after a failed health-check reconnect), try
        // to backfill one — but only once the backoff window has elapsed, so
        // a sustained outage doesn't turn into a reconnect storm against the
        // database.
        if (total_created_ < pool_size_ && now >= next_retry_at_) {
            lock.unlock();
            std::unique_ptr<pqxx::connection> fresh;
            std::string fail_reason;
            try {
                fresh = create_connection();
            } catch (const std::exception& e) {
                fail_reason = e.what();
            }
            lock.lock();

            if (fresh) {
                ++total_created_;
                retry_delay_ = kInitialRetryDelay;
                next_retry_at_ = {};
                spdlog::info("DbPool: backfilled connection ({}/{})", total_created_, pool_size_);
                conn = std::move(fresh);
                break;
            }

            spdlog::warn("DbPool: backfill attempt failed ({}), retrying in {}ms",
                         fail_reason, retry_delay_.count());
            next_retry_at_ = std::chrono::steady_clock::now() + retry_delay_;
            retry_delay_ = std::min(retry_delay_ * 2, kMaxRetryDelay);
            now = std::chrono::steady_clock::now();
        }

        if (now >= deadline) {
            throw std::runtime_error("DB pool exhausted — no connection available after 10s");
        }

        // Wake up whenever a connection is returned, our backoff window
        // opens, or the overall deadline is reached — whichever comes first.
        auto wait_until = std::min(deadline, std::max(now, next_retry_at_));
        cv_.wait_until(lock, wait_until);
    }
    lock.unlock();

    // Verify connection is still alive
    try {
        pqxx::nontransaction ntx(*conn);
        ntx.exec("SELECT 1");
    } catch (...) {
        spdlog::warn("Stale DB connection detected, reconnecting");
        try {
            conn = create_connection();
        } catch (const std::exception& e) {
            spdlog::error("Failed to reconnect: {}", e.what());
            // Drop the dead connection rather than recycling it — total_created_
            // shrinks so a future acquire() treats this as a missing slot and
            // backfills it (with backoff) once the database is reachable again.
            {
                std::lock_guard relock(mutex_);
                --total_created_;
            }
            throw;
        }
    }

    return ConnectionGuard(*this, std::move(conn));
}

void DbPool::return_connection(std::unique_ptr<pqxx::connection> conn) {
    {
        std::lock_guard lock(mutex_);
        pool_.push(std::move(conn));
    }
    cv_.notify_one();
}

DbPool::Stats DbPool::stats() const {
    std::lock_guard lock(mutex_);
    return Stats{
        .total_connections = total_created_,
        .available_connections = static_cast<int>(pool_.size()),
        .in_use_connections = total_created_ - static_cast<int>(pool_.size()),
    };
}

// ConnectionGuard implementation

DbPool::ConnectionGuard::ConnectionGuard(DbPool& pool, std::unique_ptr<pqxx::connection> conn)
    : pool_(&pool), conn_(std::move(conn)) {}

DbPool::ConnectionGuard::~ConnectionGuard() {
    if (conn_ && pool_) {
        pool_->return_connection(std::move(conn_));
    }
}

DbPool::ConnectionGuard::ConnectionGuard(ConnectionGuard&& other) noexcept
    : pool_(other.pool_), conn_(std::move(other.conn_)) {
    other.pool_ = nullptr;
}

} // namespace hms
