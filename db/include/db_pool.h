#pragma once

#include <string>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <queue>
#include <pqxx/pqxx>

namespace hms {

/// Thread-safe PostgreSQL connection pool using libpqxx
class DbPool {
public:
    struct Config {
        std::string host = "localhost";
        int port = 5432;
        std::string user = "dbuser";
        std::string password;
        std::string database = "ai_context";
        int pool_size = 4;
    };

    explicit DbPool(const Config& config);
    ~DbPool();

    // Non-copyable, non-movable
    DbPool(const DbPool&) = delete;
    DbPool& operator=(const DbPool&) = delete;

    /// RAII connection guard - returns connection to pool on destruction
    class ConnectionGuard {
    public:
        ConnectionGuard(DbPool& pool, std::unique_ptr<pqxx::connection> conn);
        ~ConnectionGuard();

        ConnectionGuard(ConnectionGuard&&) noexcept;
        ConnectionGuard& operator=(ConnectionGuard&&) = delete;
        ConnectionGuard(const ConnectionGuard&) = delete;
        ConnectionGuard& operator=(const ConnectionGuard&) = delete;

        pqxx::connection& operator*() { return *conn_; }
        pqxx::connection* operator->() { return conn_.get(); }

    private:
        DbPool* pool_;
        std::unique_ptr<pqxx::connection> conn_;
    };

    /// Acquire a connection from the pool (blocks if none available).
    /// If the pool never reached its configured size — e.g. the database was
    /// unreachable when some initial connections were created — acquire()
    /// will opportunistically try to backfill a missing slot, subject to
    /// exponential backoff so a sustained outage doesn't turn into a
    /// reconnect storm against the database.
    ConnectionGuard acquire();

    /// Get the connection string
    const std::string& connection_string() const { return conn_string_; }

    /// Get pool statistics
    struct Stats {
        int total_connections;
        int available_connections;
        int in_use_connections;
    };
    Stats stats() const;

private:
    void return_connection(std::unique_ptr<pqxx::connection> conn);
    std::unique_ptr<pqxx::connection> create_connection();

    std::string conn_string_;
    int pool_size_;
    std::queue<std::unique_ptr<pqxx::connection>> pool_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    int total_created_ = 0;

    // Backfill backoff state (guarded by mutex_). When the pool is short of
    // pool_size_ connections, acquire() won't retry creating one again until
    // steady_clock::now() >= next_retry_at_. retry_delay_ doubles on every
    // failed backfill attempt, up to kMaxRetryDelay, and resets to
    // kInitialRetryDelay as soon as a backfill succeeds.
    static constexpr std::chrono::milliseconds kInitialRetryDelay{1000};
    static constexpr std::chrono::milliseconds kMaxRetryDelay{30000};
    std::chrono::steady_clock::time_point next_retry_at_{};
    std::chrono::milliseconds retry_delay_{kInitialRetryDelay};
};

} // namespace hms
