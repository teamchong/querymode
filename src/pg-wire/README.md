# pg-wire/ — PostgreSQL Wire Protocol

> Connect to QueryMode with `psql`, DBeaver, Metabase, or any PostgreSQL client.
> Implements the PostgreSQL v3 wire protocol over TCP or WebSocket.

## Architecture

```
psql / BI tool
  │
  │  PostgreSQL wire protocol (binary)
  ▼
protocol.ts ──► Parse binary messages (Query, Parse, Bind, Execute, Describe)
  │
  ▼
handler.ts ──► PgConnectionHandler
  │              Routes simple queries → sql/executor.ts
  │              Routes extended queries → Parse/Bind/Execute cycle
  │              Manages session state (prepared statements, portals)
  │
  ▼
server.ts ──► TCP server wrapper (Node/Bun)
               Accepts connections, spawns handlers
```

## Files

| File | Purpose |
|------|---------|
| protocol.ts | Binary message encoding/decoding. Startup, Query, RowDescription, DataRow, etc. |
| handler.ts | Connection lifecycle. Auth (trust/md5), query dispatch, error handling, cancellation |
| server.ts | TCP listener. One handler per connection |

## Supported Protocol Features

- **Simple query protocol**: `Query` message → parse SQL → execute → stream rows
- **Extended query protocol**: `Parse` → `Bind` → `Describe` → `Execute` → `Sync`
- **Type mapping**: INT8, FLOAT8, TEXT, BOOL, BYTEA, TIMESTAMP
- **Auth**: Trust (no password) or MD5 password
- **Cancellation**: `CancelRequest` terminates in-flight queries
- **SSL**: TLS negotiation (reject or upgrade)

## Usage

```typescript
import { PgConnectionHandler } from "querymode"

// Each TCP connection gets a handler
const handler = new PgConnectionHandler({
  executor: queryMode.getExecutor(),
  onQuery: (sql) => console.log("Query:", sql),
})
```

```bash
# Connect with psql
psql -h localhost -p 5433 -U querymode

# Run queries
SELECT * FROM my_table WHERE status = 'active' LIMIT 10;
SELECT category, COUNT(*) FROM sales GROUP BY category;
```

## Limitations

- No transactions (every query auto-commits)
- No cursors (full result set in memory)
- No COPY protocol
- No advisory locks or LISTEN/NOTIFY
