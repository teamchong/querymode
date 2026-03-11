/**
 * Multi-stage ETL pipeline — the Spark replacement.
 *
 * Each stage executes a query, writes intermediate results to R2,
 * and feeds them to the next stage. TypeScript is the DAG scheduler.
 *
 * This example: raw events → click aggregation → high-traffic filter → top pages
 */
import { QueryMode, Pipeline } from "querymode";

// Edge mode: DOs + R2
const qm = QueryMode.remote(env.QUERY_DO, {
  region: "SJC",
  masterDoNamespace: env.MASTER_DO,
});

// Option A: Pipeline class (auto-naming, auto-cleanup)
const result = await Pipeline.create(qm.table("events"))
  .stage(df => df
    .filter("type", "eq", "click")
    .groupBy("page")
    .aggregate("count", "*")
    .aggregate("avg", "duration"))
  .stage(df => df
    .filter("count_*", "gt", 100)
    .sort("avg_duration", "desc")
    .limit(50))
  .run();

console.log("Top 50 high-traffic slow pages:", result.rows);

// Option B: Manual stages with materializeAs (full control)
const clicksByPage = await qm.table("events")
  .filter("type", "eq", "click")
  .groupBy("page")
  .aggregate("count", "*")
  .materializeAs("clicks_by_page");

const topPages = await clicksByPage
  .filter("count_*", "gt", 100)
  .sort("count_*", "desc")
  .limit(50)
  .collect();

// Cleanup intermediate table
await qm.table("clicks_by_page").dropTable();

console.log("Top pages:", topPages.rows);

// Option C: Complex DAG with branching
const dailyAgg = await qm.table("events")
  .filter("date", "eq", "2026-03-11")
  .groupBy("page", "country")
  .aggregate("count", "*")
  .materializeAs("daily_agg");

// Branch 1: top pages by country
const topByCountry = await dailyAgg
  .groupBy("country")
  .aggregate("sum", "count_*")
  .sort("sum_count_*", "desc")
  .collect();

// Branch 2: pages with anomalous traffic
const anomalies = await dailyAgg
  .filter("count_*", "gt", 10000)
  .collect();

await qm.table("daily_agg").dropTable();

console.log("Top countries:", topByCountry.rows);
console.log("Anomalies:", anomalies.rows);
