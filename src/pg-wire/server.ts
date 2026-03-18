/**
 * Local Postgres wire protocol server — connect with psql or any BI tool.
 *
 * Usage:
 *   npx tsx src/pg-wire/server.ts [path-to-data]
 *
 *   psql -h localhost -p 5433 -U querymode -d data
 *   SELECT * FROM events WHERE region = 'us' LIMIT 10;
 */

import * as net from "node:net";
import { QueryMode } from "../local.js";
import { PgConnectionHandler } from "./handler.js";

const PORT = parseInt(process.env.PG_PORT ?? "5433", 10);
const HOST = process.env.PG_HOST ?? "127.0.0.1";

const qm = QueryMode.local();

const server = net.createServer((socket) => {
  const handler = new PgConnectionHandler({
    executor: qm.getExecutor(),
    send: (data) => {
      if (!socket.destroyed) socket.write(data);
    },
  });

  socket.on("data", async (chunk: Buffer) => {
    try {
      await handler.onData(new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength));
    } catch (err) {
      console.error("Protocol error:", err);
      socket.destroy();
    }
  });

  socket.on("error", () => {});
  socket.on("close", () => {});
});

server.listen(PORT, HOST, () => {
  console.log(`QueryMode Postgres wire server listening on ${HOST}:${PORT}`);
  console.log(`Connect with: psql -h ${HOST} -p ${PORT} -U querymode`);
  console.log(`Query files:  SELECT * FROM './data/events.parquet' LIMIT 10;`);
});
