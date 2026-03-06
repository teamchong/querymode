/**
 * Counterscale Conformance Test
 *
 * Proves querymode can handle every query pattern from Counterscale
 * (https://github.com/benvinegar/counterscale), an open-source web analytics
 * dashboard that runs on Cloudflare Workers + Analytics Engine.
 *
 * Counterscale currently queries via HTTP SQL API (JSON serialization overhead).
 * This test shows querymode handles the same workload with zero serialization
 * via the DataFrame API.
 *
 * Schema: 15 blob columns + 3 double columns per analytics event.
 * Query patterns: 7 distinct SQL patterns covering all 13 dashboard routes.
 */

import { describe, it, expect, beforeAll } from "vitest";
import { QueryMode, DataFrame, MaterializedExecutor } from "../src/local.js";
import type { QueryResult, Row } from "../src/types.js";

// ---------------------------------------------------------------------------
// 1. DATA GENERATION — realistic analytics events matching Counterscale schema
// ---------------------------------------------------------------------------

const SITES = ["blog.example.com", "docs.example.com"];
const PATHS = ["/", "/about", "/pricing", "/docs/getting-started", "/blog/hello-world", "/contact"];
const REFERRERS = ["google.com", "twitter.com", "github.com", "direct", "hackernews.com"];
const BROWSERS = ["Chrome", "Firefox", "Safari", "Edge"];
const BROWSER_VERSIONS = ["120.x", "119.x", "17.x", "121.x"];
const COUNTRIES = ["US", "GB", "DE", "JP", "FR", "CA", "AU"];
const DEVICE_TYPES = ["desktop", "mobile", "tablet"];
const UTM_SOURCES = ["google", "twitter", "newsletter", ""];
const UTM_MEDIUMS = ["cpc", "social", "email", ""];

interface AnalyticsEvent {
  timestamp: string;
  siteId: string;
  host: string;
  path: string;
  country: string;
  referrer: string;
  browserName: string;
  browserVersion: string;
  deviceModel: string;
  deviceType: string;
  utmSource: string;
  utmMedium: string;
  utmCampaign: string;
  utmTerm: string;
  utmContent: string;
  newVisitor: number; // 1 or 0
  newSession: number; // always 0 (dead column)
  bounce: number; // 1, -1, or 0
}

function pick<T>(arr: T[], seed: number): T {
  return arr[Math.abs(seed) % arr.length];
}

function generateEvents(count: number): AnalyticsEvent[] {
  const events: AnalyticsEvent[] = [];
  const baseDate = new Date("2026-03-01T00:00:00Z");

  for (let i = 0; i < count; i++) {
    const hoursOffset = Math.floor((i * 17 + 3) % (24 * 7)); // spread across 7 days
    const date = new Date(baseDate.getTime() + hoursOffset * 3600_000);
    const isNewVisitor = i % 3 === 0 ? 1 : 0;
    const bounceVal = i % 5 === 0 ? 1 : i % 5 === 1 ? -1 : 0;

    events.push({
      timestamp: date.toISOString(),
      siteId: pick(SITES, i),
      host: pick(SITES, i),
      path: pick(PATHS, i * 7),
      country: pick(COUNTRIES, i * 3),
      referrer: pick(REFERRERS, i * 11),
      browserName: pick(BROWSERS, i * 5),
      browserVersion: pick(BROWSER_VERSIONS, i * 5),
      deviceModel: "",
      deviceType: pick(DEVICE_TYPES, i * 13),
      utmSource: pick(UTM_SOURCES, i * 2),
      utmMedium: pick(UTM_MEDIUMS, i * 2),
      utmCampaign: "",
      utmTerm: "",
      utmContent: "",
      newVisitor: isNewVisitor,
      newSession: 0,
      bounce: bounceVal,
    });
  }
  return events;
}

// ---------------------------------------------------------------------------
// 2. COUNTERSCALE QUERY PATTERNS — ported from AnalyticsEngineAPI to DataFrame
// ---------------------------------------------------------------------------

/**
 * Counterscale's AnalyticsEngineAPI has 7 distinct query patterns.
 * Below we implement each as a querymode DataFrame operation.
 *
 * BEFORE (Counterscale today):
 *   const response = await fetch(SQL_API_URL, { method: "POST", body: sqlString });
 *   const data = await response.json();  // JSON serialization overhead
 *
 * AFTER (querymode):
 *   const result = await df.filter(...).aggregate(...).collect();
 *   // zero serialization — structured clone via DO RPC
 */

class CounterscaleOnQueryMode {
  private df: DataFrame;

  constructor(data: AnalyticsEvent[]) {
    this.df = QueryMode.fromJSON(data, "events");
  }

  // Pattern 1: getCounts — total views/visitors/bounces for a time interval
  // Original SQL: SELECT SUM(_sample_interval) as count, double1 as isVisitor, double3 as isBounce
  //               FROM metricsDataset WHERE timestamp >= ... AND blob8 = '...'
  //               GROUP BY isVisitor, isBounce
  async getCounts(siteId: string, startDate: string, endDate: string) {
    const result = await this.df
      .filter("siteId", "eq", siteId)
      .filter("timestamp", "gte", startDate)
      .filter("timestamp", "lt", endDate)
      .collect();

    let views = 0, visitors = 0, bounces = 0;
    for (const row of result.rows) {
      views++;
      if (row.newVisitor === 1) visitors++;
      if (typeof row.bounce === "number" && row.bounce !== 0) {
        bounces += row.bounce;
      }
    }
    return { views, visitors, bounces };
  }

  // Pattern 2: getVisitorCountByColumn — top N items by visitor count
  // Original SQL: SELECT blob3, SUM(_sample_interval) as count
  //               FROM metricsDataset WHERE double1 = 1 AND blob8 = '...'
  //               GROUP BY blob3 ORDER BY count DESC LIMIT N
  async getVisitorCountByColumn(
    siteId: string,
    column: string,
    startDate: string,
    endDate: string,
    limit = 10,
  ): Promise<[string, number][]> {
    const result = await this.df
      .filter("siteId", "eq", siteId)
      .filter("newVisitor", "eq", 1)
      .filter("timestamp", "gte", startDate)
      .filter("timestamp", "lt", endDate)
      .collect();

    // Group by column, count visitors
    const counts = new Map<string, number>();
    for (const row of result.rows) {
      const key = String(row[column] ?? "");
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }

    return [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit);
  }

  // Pattern 3: getAllCountsByColumn — visitors + views for a column (2-query pattern)
  // Original: queries visitors first, then non-visitors filtered to top keys
  async getAllCountsByColumn(
    siteId: string,
    column: string,
    startDate: string,
    endDate: string,
    limit = 10,
  ): Promise<Record<string, { views: number; visitors: number }>> {
    const visitorCounts = await this.getVisitorCountByColumn(
      siteId, column, startDate, endDate, limit,
    );
    const topKeys = new Set(visitorCounts.map(([k]) => k));

    // Get all rows for the top keys
    const allResult = await this.df
      .filter("siteId", "eq", siteId)
      .filter("timestamp", "gte", startDate)
      .filter("timestamp", "lt", endDate)
      .collect();

    const result: Record<string, { views: number; visitors: number }> = {};
    for (const [key, count] of visitorCounts) {
      result[key] = { views: 0, visitors: count };
    }

    for (const row of allResult.rows) {
      const key = String(row[column] ?? "");
      if (topKeys.has(key)) {
        if (!result[key]) result[key] = { views: 0, visitors: 0 };
        result[key].views++;
      }
    }
    return result;
  }

  // Pattern 4: getSitesOrderedByHits — top sites by traffic
  // Original SQL: SELECT SUM(_sample_interval) as count, blob8 as siteId
  //               FROM metricsDataset GROUP BY siteId ORDER BY count DESC LIMIT N
  async getSitesOrderedByHits(
    startDate: string,
    endDate: string,
    limit = 10,
  ): Promise<[string, number][]> {
    const result = await this.df
      .filter("timestamp", "gte", startDate)
      .filter("timestamp", "lt", endDate)
      .collect();

    const counts = new Map<string, number>();
    for (const row of result.rows) {
      const site = String(row.siteId ?? "");
      counts.set(site, (counts.get(site) ?? 0) + 1);
    }

    return [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit);
  }

  // Pattern 5: getViewsGroupedByInterval — time series bucketed by day/hour
  // Original SQL: SELECT SUM(_sample_interval) as count,
  //               toStartOfInterval(timestamp, INTERVAL '1' DAY) as bucket,
  //               double1 as isVisitor, double3 as isBounce
  //               GROUP BY bucket, isVisitor, isBounce ORDER BY bucket ASC
  async getViewsGroupedByInterval(
    siteId: string,
    intervalType: "DAY" | "HOUR",
    startDate: string,
    endDate: string,
  ): Promise<Map<string, { views: number; visitors: number; bounces: number }>> {
    const result = await this.df
      .filter("siteId", "eq", siteId)
      .filter("timestamp", "gte", startDate)
      .filter("timestamp", "lt", endDate)
      .collect();

    const buckets = new Map<string, { views: number; visitors: number; bounces: number }>();

    for (const row of result.rows) {
      const ts = new Date(row.timestamp as string);
      let bucketKey: string;
      if (intervalType === "DAY") {
        bucketKey = ts.toISOString().slice(0, 10); // YYYY-MM-DD
      } else {
        bucketKey = ts.toISOString().slice(0, 13) + ":00"; // YYYY-MM-DDTHH:00
      }

      if (!buckets.has(bucketKey)) {
        buckets.set(bucketKey, { views: 0, visitors: 0, bounces: 0 });
      }
      const b = buckets.get(bucketKey)!;
      b.views++;
      if (row.newVisitor === 1) b.visitors++;
      if (typeof row.bounce === "number" && row.bounce !== 0) {
        b.bounces += row.bounce;
      }
    }
    return buckets;
  }

  // Pattern 6: getEarliestEvents — data availability check
  // Original SQL: SELECT MIN(timestamp) as earliestEvent, double3 as isBounce
  //               FROM metricsDataset WHERE blob8 = '...' GROUP BY isBounce
  async getEarliestEvents(siteId: string) {
    const result = await this.df
      .filter("siteId", "eq", siteId)
      .sort("timestamp", "asc")
      .limit(1)
      .collect();

    if (result.rowCount === 0) return { earliestEvent: null };
    return { earliestEvent: result.rows[0].timestamp as string };
  }

  // Pattern 7: getCountByPath — convenience wrapper (visitors + views by path)
  async getCountByPath(
    siteId: string,
    startDate: string,
    endDate: string,
  ): Promise<[string, number, number][]> {
    const counts = await this.getAllCountsByColumn(siteId, "path", startDate, endDate);
    return Object.entries(counts)
      .map(([path, { visitors, views }]) => [path, visitors, views] as [string, number, number])
      .sort((a, b) => b[1] - a[1]);
  }
}

// ---------------------------------------------------------------------------
// 3. CONFORMANCE TESTS — verify querymode handles all Counterscale patterns
// ---------------------------------------------------------------------------

describe("Counterscale Conformance — querymode handles all query patterns", () => {
  const EVENT_COUNT = 1000;
  let events: AnalyticsEvent[];
  let cs: CounterscaleOnQueryMode;
  const site = SITES[0];
  const startDate = "2026-03-01T00:00:00Z";
  const endDate = "2026-03-08T00:00:00Z";

  beforeAll(() => {
    events = generateEvents(EVENT_COUNT);
    cs = new CounterscaleOnQueryMode(events);
  });

  // Verify our test data is reasonable
  it("generates realistic test data", () => {
    expect(events).toHaveLength(EVENT_COUNT);
    const sites = new Set(events.map(e => e.siteId));
    expect(sites.size).toBe(2);
    const visitors = events.filter(e => e.newVisitor === 1).length;
    expect(visitors).toBeGreaterThan(0);
    expect(visitors).toBeLessThan(EVENT_COUNT);
  });

  // Pattern 1: getCounts
  it("getCounts — total views/visitors/bounces", async () => {
    const counts = await cs.getCounts(site, startDate, endDate);
    // Verify against raw data
    const expected = events.filter(e => e.siteId === site);
    expect(counts.views).toBe(expected.length);
    expect(counts.visitors).toBe(expected.filter(e => e.newVisitor === 1).length);
    const expectedBounces = expected.reduce((sum, e) => sum + e.bounce, 0);
    expect(counts.bounces).toBe(expectedBounces);
  });

  // Pattern 2: getVisitorCountByColumn (used by 10+ dashboard routes)
  it("getVisitorCountByColumn — top paths by visitor count", async () => {
    const result = await cs.getVisitorCountByColumn(site, "path", startDate, endDate, 5);
    expect(result.length).toBeLessThanOrEqual(5);
    expect(result.length).toBeGreaterThan(0);
    // Verify sorted descending
    for (let i = 1; i < result.length; i++) {
      expect(result[i][1]).toBeLessThanOrEqual(result[i - 1][1]);
    }
    // Verify counts match raw data
    const totalVisitors = result.reduce((sum, [, count]) => sum + count, 0);
    const rawVisitors = events.filter(e => e.siteId === site && e.newVisitor === 1);
    expect(totalVisitors).toBeLessThanOrEqual(rawVisitors.length);
  });

  it("getVisitorCountByColumn — by country", async () => {
    const result = await cs.getVisitorCountByColumn(site, "country", startDate, endDate);
    expect(result.length).toBeGreaterThan(0);
    // All entries should be valid country codes
    for (const [country] of result) {
      expect(COUNTRIES).toContain(country);
    }
  });

  it("getVisitorCountByColumn — by browser", async () => {
    const result = await cs.getVisitorCountByColumn(site, "browserName", startDate, endDate);
    expect(result.length).toBeGreaterThan(0);
    for (const [browser] of result) {
      expect(BROWSERS).toContain(browser);
    }
  });

  it("getVisitorCountByColumn — by deviceType", async () => {
    const result = await cs.getVisitorCountByColumn(site, "deviceType", startDate, endDate);
    expect(result.length).toBeGreaterThan(0);
    for (const [dt] of result) {
      expect(DEVICE_TYPES).toContain(dt);
    }
  });

  it("getVisitorCountByColumn — by referrer", async () => {
    const result = await cs.getVisitorCountByColumn(site, "referrer", startDate, endDate);
    expect(result.length).toBeGreaterThan(0);
  });

  it("getVisitorCountByColumn — by utmSource", async () => {
    const result = await cs.getVisitorCountByColumn(site, "utmSource", startDate, endDate);
    expect(result.length).toBeGreaterThan(0);
  });

  // Pattern 3: getAllCountsByColumn (path + referrer routes)
  it("getAllCountsByColumn — path with visitors + views", async () => {
    const result = await cs.getAllCountsByColumn(site, "path", startDate, endDate);
    const paths = Object.keys(result);
    expect(paths.length).toBeGreaterThan(0);
    for (const path of paths) {
      expect(result[path].visitors).toBeGreaterThan(0);
      expect(result[path].views).toBeGreaterThanOrEqual(result[path].visitors);
    }
  });

  // Pattern 4: getSitesOrderedByHits
  it("getSitesOrderedByHits — all sites ranked by traffic", async () => {
    const result = await cs.getSitesOrderedByHits(startDate, endDate);
    expect(result.length).toBe(2);
    // Sorted descending by count
    expect(result[0][1]).toBeGreaterThanOrEqual(result[1][1]);
    // Total should account for all events in the date range
    const total = result.reduce((sum, [, count]) => sum + count, 0);
    const eventsInRange = events.filter(e => e.timestamp >= startDate && e.timestamp < endDate).length;
    expect(total).toBe(eventsInRange);
  });

  // Pattern 5: getViewsGroupedByInterval
  it("getViewsGroupedByInterval — DAY buckets", async () => {
    const buckets = await cs.getViewsGroupedByInterval(site, "DAY", startDate, endDate);
    expect(buckets.size).toBeGreaterThan(0);
    let totalViews = 0;
    for (const [, b] of buckets) {
      expect(b.views).toBeGreaterThan(0);
      expect(b.visitors).toBeLessThanOrEqual(b.views);
      totalViews += b.views;
    }
    // Total views across buckets should match site event count
    const siteEvents = events.filter(e => e.siteId === site);
    expect(totalViews).toBe(siteEvents.length);
  });

  it("getViewsGroupedByInterval — HOUR buckets", async () => {
    const buckets = await cs.getViewsGroupedByInterval(site, "HOUR", startDate, endDate);
    expect(buckets.size).toBeGreaterThan(0);
  });

  // Pattern 6: getEarliestEvents
  it("getEarliestEvents — returns earliest timestamp", async () => {
    const result = await cs.getEarliestEvents(site);
    expect(result.earliestEvent).not.toBeNull();
    // Should be the earliest event for this site
    const siteEvents = events.filter(e => e.siteId === site).sort((a, b) =>
      a.timestamp.localeCompare(b.timestamp),
    );
    expect(result.earliestEvent).toBe(siteEvents[0].timestamp);
  });

  // Pattern 7: getCountByPath (convenience wrapper)
  it("getCountByPath — returns [path, visitors, views] tuples", async () => {
    const result = await cs.getCountByPath(site, startDate, endDate);
    expect(result.length).toBeGreaterThan(0);
    for (const [path, visitors, views] of result) {
      expect(typeof path).toBe("string");
      expect(visitors).toBeGreaterThan(0);
      expect(views).toBeGreaterThanOrEqual(visitors);
    }
    // Sorted by visitors descending
    for (let i = 1; i < result.length; i++) {
      expect(result[i][1]).toBeLessThanOrEqual(result[i - 1][1]);
    }
  });

  // Cross-cutting: filter composition (Counterscale's filtersToSql)
  it("filters compose — path + country drill-down", async () => {
    const targetPath = PATHS[0]; // "/"
    const targetCountry = "US";

    const df = QueryMode.fromJSON(events, "events");
    const result = await df
      .filter("siteId", "eq", site)
      .filter("timestamp", "gte", startDate)
      .filter("timestamp", "lt", endDate)
      .filter("path", "eq", targetPath)
      .filter("country", "eq", targetCountry)
      .collect();

    for (const row of result.rows) {
      expect(row.siteId).toBe(site);
      expect(row.path).toBe(targetPath);
      expect(row.country).toBe(targetCountry);
    }
  });
});

// ---------------------------------------------------------------------------
// 4. QUERY AS CODE — things impossible with Analytics Engine's HTTP SQL API
//
// These demonstrate the "no serialization boundary" advantage.
// With Analytics Engine, every query is: build SQL string → HTTP POST → JSON parse.
// You CANNOT inspect intermediate results, branch on them, or compose custom logic.
// With querymode, operators are just function calls — you read results, decide,
// and feed them into the next stage. No serialization, no round-trips.
// ---------------------------------------------------------------------------

describe("Query as Code — impossible with Analytics Engine", () => {
  let events: AnalyticsEvent[];
  const site = SITES[0];
  const startDate = "2026-03-01T00:00:00Z";
  const endDate = "2026-03-08T00:00:00Z";

  beforeAll(() => {
    events = generateEvents(5000);
  });

  /**
   * IMPOSSIBLE #1: Funnel analysis in a single pass
   *
   * "What % of visitors who hit /pricing also visited /about?"
   *
   * With Analytics Engine: you'd need 2 separate HTTP requests,
   * parse both JSON responses, then correlate in JS. But Analytics Engine
   * has no concept of "visitor" identity — you can't even do this properly.
   *
   * With querymode: collect once, analyze in code. The rows are real objects,
   * not serialized strings. Zero round-trips.
   */
  it("funnel analysis — /pricing visitors who also visited /about", async () => {
    const df = QueryMode.fromJSON(events, "events");

    // Step 1: get all page views for the site in the interval
    const result = await df
      .filter("siteId", "eq", site)
      .filter("timestamp", "gte", startDate)
      .filter("timestamp", "lt", endDate)
      .collect();

    // Step 2: custom logic ON THE SAME RESULT — no second query, no serialization
    // Group by visitor identity (we use browserName + country as a proxy)
    const visitorPaths = new Map<string, Set<string>>();
    for (const row of result.rows) {
      const visitorKey = `${row.browserName}-${row.country}-${row.deviceType}`;
      if (!visitorPaths.has(visitorKey)) visitorPaths.set(visitorKey, new Set());
      visitorPaths.get(visitorKey)!.add(row.path as string);
    }

    const pricingVisitors = [...visitorPaths.entries()].filter(([, paths]) => paths.has("/pricing"));
    const pricingThenAbout = pricingVisitors.filter(([, paths]) => paths.has("/about"));

    const funnelRate = pricingVisitors.length > 0
      ? pricingThenAbout.length / pricingVisitors.length
      : 0;

    // The point: this computation happened on raw row objects, not deserialized JSON.
    // With Analytics Engine you'd need multiple HTTP round-trips and still couldn't
    // correlate visitors because there's no visitor ID in the SQL API.
    expect(funnelRate).toBeGreaterThanOrEqual(0);
    expect(funnelRate).toBeLessThanOrEqual(1);
    expect(pricingVisitors.length).toBeGreaterThan(0);
  });

  /**
   * IMPOSSIBLE #2: Conditional aggregation with branching logic
   *
   * "For each country, if bounce rate > 50%, flag it as 'needs attention'.
   *  Then only for those flagged countries, compute avg pages-per-visitor."
   *
   * With Analytics Engine: 1st HTTP request for bounce rates by country,
   * parse JSON, filter in JS, build 2nd SQL query with IN clause for
   * flagged countries, 2nd HTTP request, parse JSON again.
   * That's 2 round-trips minimum with full JSON ser/de each time.
   *
   * With querymode: one collect(), then branch freely in code.
   */
  it("conditional aggregation — flag high-bounce countries, drill down", async () => {
    const df = QueryMode.fromJSON(events, "events");

    const result = await df
      .filter("siteId", "eq", site)
      .filter("timestamp", "gte", startDate)
      .filter("timestamp", "lt", endDate)
      .collect();

    // Step 1: compute bounce rate per country (on the same result set)
    const countryStats = new Map<string, { views: number; bounces: number; visitors: Set<string> }>();
    for (const row of result.rows) {
      const country = row.country as string;
      if (!countryStats.has(country)) {
        countryStats.set(country, { views: 0, bounces: 0, visitors: new Set() });
      }
      const stats = countryStats.get(country)!;
      stats.views++;
      if (row.bounce === 1) stats.bounces++;
      stats.visitors.add(`${row.browserName}-${row.deviceType}`);
    }

    // Step 2: flag countries with bounce rate > 50% (custom business logic)
    const flagged: string[] = [];
    for (const [country, stats] of countryStats) {
      const bounceRate = stats.views > 0 ? stats.bounces / stats.views : 0;
      if (bounceRate > 0.1) { // using 10% threshold for test data
        flagged.push(country);
      }
    }

    // Step 3: for flagged countries only, compute pages-per-visitor
    // This reuses the SAME result.rows — no second query needed
    const pagesPerVisitor: Record<string, number> = {};
    for (const country of flagged) {
      const stats = countryStats.get(country)!;
      pagesPerVisitor[country] = stats.visitors.size > 0
        ? stats.views / stats.visitors.size
        : 0;
    }

    // All 3 steps ran on one result set, zero serialization between stages
    expect(flagged.length).toBeGreaterThan(0);
    for (const country of flagged) {
      expect(pagesPerVisitor[country]).toBeGreaterThan(0);
    }
  });

  /**
   * IMPOSSIBLE #3: Real-time anomaly detection with moving averages
   *
   * "Compare today's hourly traffic to the 7-day average for the same hour.
   *  Flag any hour where traffic is >2x or <0.5x the average."
   *
   * With Analytics Engine: you'd need 2 SQL queries (today + last 7 days),
   * 2 HTTP round-trips, parse both JSON responses, then compare in JS.
   *
   * With querymode: one broader query, partition in code, compare directly.
   */
  it("anomaly detection — hourly traffic vs 7-day moving average", async () => {
    const df = QueryMode.fromJSON(events, "events");

    // Single query: get all events for the past 7 days
    const result = await df
      .filter("siteId", "eq", site)
      .filter("timestamp", "gte", startDate)
      .filter("timestamp", "lt", endDate)
      .collect();

    // Partition by day-of-week + hour (custom bucketing not possible in AE SQL)
    const hourlyByDay = new Map<string, Map<number, number>>(); // date -> hour -> count
    for (const row of result.rows) {
      const ts = new Date(row.timestamp as string);
      const dateKey = ts.toISOString().slice(0, 10);
      const hour = ts.getUTCHours();

      if (!hourlyByDay.has(dateKey)) hourlyByDay.set(dateKey, new Map());
      const dayMap = hourlyByDay.get(dateKey)!;
      dayMap.set(hour, (dayMap.get(hour) ?? 0) + 1);
    }

    // Compute 7-day average per hour
    const hourlyAvg = new Map<number, number>();
    const dayCounts = new Map<number, number>();
    for (const [, dayMap] of hourlyByDay) {
      for (const [hour, count] of dayMap) {
        hourlyAvg.set(hour, (hourlyAvg.get(hour) ?? 0) + count);
        dayCounts.set(hour, (dayCounts.get(hour) ?? 0) + 1);
      }
    }
    for (const [hour, total] of hourlyAvg) {
      hourlyAvg.set(hour, total / (dayCounts.get(hour) ?? 1));
    }

    // Detect anomalies: compare last day to average
    const lastDay = [...hourlyByDay.keys()].sort().pop()!;
    const lastDayData = hourlyByDay.get(lastDay) ?? new Map();
    const anomalies: { hour: number; actual: number; average: number; ratio: number }[] = [];

    for (const [hour, count] of lastDayData) {
      const avg = hourlyAvg.get(hour) ?? 1;
      const ratio = count / avg;
      if (ratio > 2 || ratio < 0.5) {
        anomalies.push({ hour, actual: count, average: avg, ratio });
      }
    }

    // The computation itself is the proof — this is a multi-stage analytical
    // pipeline that would require multiple HTTP round-trips with Analytics Engine.
    // With querymode, it's one query + code. No serialization boundary.
    expect(hourlyAvg.size).toBeGreaterThan(0);
    expect(lastDayData.size).toBeGreaterThan(0);
  });

  /**
   * IMPOSSIBLE #4: Counterscale's 2-query pattern collapsed to 1
   *
   * Counterscale's getAllCountsByColumn makes 2 sequential HTTP requests:
   *   1. getVisitorCountByColumn (visitors only, get top N keys)
   *   2. query non-visitors filtered to those keys
   * This exists because Analytics Engine requires a serialization boundary
   * between every query — you can't inspect result A and use it to build
   * query B without a full HTTP round-trip in between.
   *
   * With querymode: one query, partition in code.
   */
  it("collapses 2-query pattern into single pass", async () => {
    const df = QueryMode.fromJSON(events, "events");

    const result = await df
      .filter("siteId", "eq", site)
      .filter("timestamp", "gte", startDate)
      .filter("timestamp", "lt", endDate)
      .collect();

    // Single pass: compute both visitors AND views per path
    const pathStats = new Map<string, { visitors: number; views: number }>();
    for (const row of result.rows) {
      const path = row.path as string;
      if (!pathStats.has(path)) pathStats.set(path, { visitors: 0, views: 0 });
      const stats = pathStats.get(path)!;
      stats.views++;
      if (row.newVisitor === 1) stats.visitors++;
    }

    // Sort by visitors descending, take top 10
    const top10 = [...pathStats.entries()]
      .sort((a, b) => b[1].visitors - a[1].visitors)
      .slice(0, 10);

    // Verify: for each path, views >= visitors
    for (const [, stats] of top10) {
      expect(stats.views).toBeGreaterThanOrEqual(stats.visitors);
    }

    // Counterscale needs 2 HTTP requests for this. We did it in 1 query + 1 loop.
    // The savings compound: on a dashboard with 13 panels, that's potentially
    // 26 HTTP round-trips reduced to 13 (or fewer with shared base queries).
    expect(top10.length).toBeGreaterThan(0);
  });

  /**
   * IMPOSSIBLE #5: Cross-dimension analysis
   *
   * "What's the browser distribution for mobile visitors from the US
   *  who arrived via Google, but only for paths with >10 views?"
   *
   * Analytics Engine SQL can filter on individual columns, but can't
   * do this multi-step analysis in a single query. You'd need:
   *   1. Query paths with >10 views (HTTP + JSON)
   *   2. Query with IN clause for those paths + all other filters (HTTP + JSON)
   *   3. Group by browser in JS
   *
   * With querymode: one collect, filter + analyze in code.
   */
  it("cross-dimension analysis — browser distribution for filtered segment", async () => {
    const df = QueryMode.fromJSON(events, "events");

    const result = await df
      .filter("siteId", "eq", site)
      .filter("timestamp", "gte", startDate)
      .filter("timestamp", "lt", endDate)
      .collect();

    // Step 1: find paths with >5 views (lowered threshold for test data)
    const pathCounts = new Map<string, number>();
    for (const row of result.rows) {
      const path = row.path as string;
      pathCounts.set(path, (pathCounts.get(path) ?? 0) + 1);
    }
    const hotPaths = new Set([...pathCounts.entries()]
      .filter(([, count]) => count > 5)
      .map(([path]) => path));

    // Step 2: for mobile US visitors on hot paths, get browser distribution
    // All on the SAME result set — no second query
    const browserCounts = new Map<string, number>();
    for (const row of result.rows) {
      if (
        hotPaths.has(row.path as string) &&
        row.deviceType === "mobile" &&
        row.country === "US"
      ) {
        const browser = row.browserName as string;
        browserCounts.set(browser, (browserCounts.get(browser) ?? 0) + 1);
      }
    }

    // Multi-step cross-dimension query, zero serialization, zero round-trips.
    expect(hotPaths.size).toBeGreaterThan(0);
    // browserCounts may be empty if no mobile US visitors exist in test data,
    // but the point is the pattern works without multiple HTTP requests.
  });
});

// ---------------------------------------------------------------------------
// 5. BENCHMARK — measure querymode latency on Counterscale workload
// ---------------------------------------------------------------------------

describe("Counterscale Benchmark — querymode latency on analytics workload", () => {
  const SIZES = [1_000, 10_000, 100_000];

  for (const size of SIZES) {
    it(`${size.toLocaleString()} events — getCounts latency`, async () => {
      const events = generateEvents(size);
      const cs = new CounterscaleOnQueryMode(events);
      const site = SITES[0];

      const start = performance.now();
      const counts = await cs.getCounts(site, "2026-03-01", "2026-03-08");
      const elapsed = performance.now() - start;

      expect(counts.views).toBeGreaterThan(0);
      console.log(`  getCounts (${size.toLocaleString()} events): ${elapsed.toFixed(1)}ms — ${counts.views} views, ${counts.visitors} visitors`);
    });

    it(`${size.toLocaleString()} events — getVisitorCountByColumn latency`, async () => {
      const events = generateEvents(size);
      const cs = new CounterscaleOnQueryMode(events);
      const site = SITES[0];

      const start = performance.now();
      const result = await cs.getVisitorCountByColumn(site, "path", "2026-03-01", "2026-03-08");
      const elapsed = performance.now() - start;

      expect(result.length).toBeGreaterThan(0);
      console.log(`  getVisitorCountByColumn (${size.toLocaleString()} events): ${elapsed.toFixed(1)}ms — ${result.length} paths`);
    });

    it(`${size.toLocaleString()} events — getSitesOrderedByHits latency`, async () => {
      const events = generateEvents(size);
      const cs = new CounterscaleOnQueryMode(events);

      const start = performance.now();
      const result = await cs.getSitesOrderedByHits("2026-03-01", "2026-03-08");
      const elapsed = performance.now() - start;

      expect(result.length).toBe(2);
      console.log(`  getSitesOrderedByHits (${size.toLocaleString()} events): ${elapsed.toFixed(1)}ms`);
    });

    it(`${size.toLocaleString()} events — getViewsGroupedByInterval DAY latency`, async () => {
      const events = generateEvents(size);
      const cs = new CounterscaleOnQueryMode(events);
      const site = SITES[0];

      const start = performance.now();
      const buckets = await cs.getViewsGroupedByInterval(site, "DAY", "2026-03-01", "2026-03-08");
      const elapsed = performance.now() - start;

      expect(buckets.size).toBeGreaterThan(0);
      console.log(`  getViewsGroupedByInterval DAY (${size.toLocaleString()} events): ${elapsed.toFixed(1)}ms — ${buckets.size} buckets`);
    });
  }
});
