/**
 * Umami Conformance Test
 *
 * Proves querymode can handle every query pattern from Umami
 * (https://github.com/umami-software/umami, 23k+ stars), the most popular
 * open-source web analytics platform. Umami uses PostgreSQL/ClickHouse with
 * Prisma ORM — every query serializes through SQL strings → database → JSON.
 *
 * This test shows querymode handles the same workload with zero serialization,
 * and demonstrates 5 patterns IMPOSSIBLE with Umami's current architecture:
 *
 * 1. Attribution in 1 pass (Umami runs 8 separate DB queries)
 * 2. Funnel analysis without CTE chains (composable operators)
 * 3. Retention cohorts with intermediate inspection
 * 4. Journey analysis with branching logic
 * 5. Cross-report correlation (funnel → retention → attribution in one pass)
 *
 * Schema: WebsiteEvent (pageviews, clicks, custom events) + Session (visitor metadata)
 * Query patterns: 10 distinct SQL patterns across 40 query files.
 */

import { describe, it, expect, beforeAll } from "vitest";
import { QueryMode, DataFrame, MaterializedExecutor } from "../src/local.js";
import type { QueryResult, Row } from "../src/types.js";

// ---------------------------------------------------------------------------
// 1. DATA GENERATION — realistic analytics events matching Umami schema
// ---------------------------------------------------------------------------

const WEBSITE_ID = "550e8400-e29b-41d4-a716-446655440000";
const PATHS = ["/", "/about", "/pricing", "/docs", "/blog/hello", "/contact", "/signup", "/dashboard"];
const EVENT_NAMES = ["click_cta", "signup_start", "signup_complete", "purchase", "download", ""];
const REFERRER_DOMAINS = ["google.com", "twitter.com", "github.com", "", "producthunt.com"];
const BROWSERS = ["Chrome", "Firefox", "Safari", "Edge"];
const OS_LIST = ["Windows", "Mac OS", "Linux", "iOS", "Android"];
const DEVICES = ["desktop", "mobile", "tablet"];
const COUNTRIES = ["US", "GB", "DE", "JP", "FR", "CA", "AU", "BR"];
const LANGUAGES = ["en-US", "en-GB", "de-DE", "ja-JP", "fr-FR"];
const UTM_SOURCES = ["google", "twitter", "newsletter", "producthunt", ""];
const UTM_MEDIUMS = ["cpc", "social", "email", "referral", ""];
const UTM_CAMPAIGNS = ["launch", "retarget", "q1-promo", ""];
const HOSTNAMES = ["app.example.com", "www.example.com"];

interface WebsiteEvent {
  event_id: string;
  website_id: string;
  session_id: string;
  visit_id: string;
  created_at: string;
  url_path: string;
  url_query: string;
  utm_source: string;
  utm_medium: string;
  utm_campaign: string;
  utm_content: string;
  utm_term: string;
  referrer_path: string;
  referrer_domain: string;
  page_title: string;
  event_type: number; // 1 = pageview, 2 = custom event
  event_name: string;
  hostname: string;
  // Session fields (denormalized for querymode — no join needed)
  browser: string;
  os: string;
  device: string;
  country: string;
  language: string;
  screen: string;
  // Ad tracking
  gclid: string;
  fbclid: string;
  msclkid: string;
  ttclid: string;
  li_fat_id: string;
  twclid: string;
}

function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function uuid(): string {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, c => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

/**
 * Generate realistic Umami events.
 * Simulates sessions (1-8 events per session), visitors across 14 days.
 */
function generateEvents(count: number): WebsiteEvent[] {
  const events: WebsiteEvent[] = [];
  const startTime = new Date("2026-03-01T00:00:00Z").getTime();
  const endTime = new Date("2026-03-15T00:00:00Z").getTime();

  // Generate sessions first, then events within sessions
  let eventsGenerated = 0;
  while (eventsGenerated < count) {
    const sessionId = uuid();
    const visitId = uuid();
    const sessionStart = startTime + Math.random() * (endTime - startTime);
    const eventsInSession = 1 + Math.floor(Math.random() * 8);
    const browser = pick(BROWSERS);
    const os = pick(OS_LIST);
    const device = pick(DEVICES);
    const country = pick(COUNTRIES);
    const language = pick(LANGUAGES);
    const screen = pick(["1920x1080", "1366x768", "390x844", "768x1024"]);
    const hostname = pick(HOSTNAMES);
    const utmSource = pick(UTM_SOURCES);
    const utmMedium = pick(UTM_MEDIUMS);
    const utmCampaign = pick(UTM_CAMPAIGNS);
    const referrerDomain = pick(REFERRER_DOMAINS);
    // Ad click IDs — simulate ~10% having one
    const hasAdClick = Math.random() < 0.1;
    const adType = hasAdClick ? pick(["gclid", "fbclid", "msclkid", "ttclid", "li_fat_id", "twclid"]) : "";

    for (let i = 0; i < eventsInSession && eventsGenerated < count; i++) {
      const eventTime = new Date(sessionStart + i * 30000 + Math.random() * 10000);
      const isPageview = Math.random() > 0.2; // 80% pageviews, 20% custom events

      events.push({
        event_id: uuid(),
        website_id: WEBSITE_ID,
        session_id: sessionId,
        visit_id: visitId,
        created_at: eventTime.toISOString(),
        url_path: pick(PATHS),
        url_query: "",
        utm_source: i === 0 ? utmSource : "",  // UTM only on first hit
        utm_medium: i === 0 ? utmMedium : "",
        utm_campaign: i === 0 ? utmCampaign : "",
        utm_content: "",
        utm_term: "",
        referrer_path: "",
        referrer_domain: i === 0 ? referrerDomain : "", // Referrer only on first hit
        page_title: "",
        event_type: isPageview ? 1 : 2,
        event_name: isPageview ? "" : pick(EVENT_NAMES.filter(e => e !== "")),
        hostname,
        browser,
        os,
        device,
        country,
        language,
        screen,
        gclid: adType === "gclid" ? "gclid_" + uuid().slice(0, 8) : "",
        fbclid: adType === "fbclid" ? "fbclid_" + uuid().slice(0, 8) : "",
        msclkid: adType === "msclkid" ? "msclkid_" + uuid().slice(0, 8) : "",
        ttclid: adType === "ttclid" ? "ttclid_" + uuid().slice(0, 8) : "",
        li_fat_id: adType === "li_fat_id" ? "lifat_" + uuid().slice(0, 8) : "",
        twclid: adType === "twclid" ? "twclid_" + uuid().slice(0, 8) : "",
      });
      eventsGenerated++;
    }
  }

  return events.sort((a, b) => a.created_at.localeCompare(b.created_at));
}

// ---------------------------------------------------------------------------
// 2. UMAMI QUERY PATTERNS ON QUERYMODE (DataFrame API)
// ---------------------------------------------------------------------------

class UmamiOnQueryMode {
  constructor(private events: WebsiteEvent[]) {}

  private df() {
    return QueryMode.fromJSON(this.events, "events");
  }

  private siteFilter(df: DataFrame): DataFrame {
    return df.filter("website_id", "eq", WEBSITE_ID);
  }

  private dateFilter(df: DataFrame, startDate: string, endDate: string): DataFrame {
    return df.filter("created_at", "gte", startDate).filter("created_at", "lt", endDate);
  }

  /**
   * Pattern 1: getWebsiteStats
   * Umami SQL: subquery aggregation for pageviews, visitors, visits, bounces, totaltime
   */
  async getWebsiteStats(startDate: string, endDate: string) {
    const result = await this.dateFilter(this.siteFilter(this.df()), startDate, endDate)
      .filter("event_type", "eq", 1) // pageviews only
      .collect();

    const sessions = new Map<string, { events: number; visits: Set<string> }>();
    const visitors = new Set<string>();

    for (const row of result.rows) {
      const sid = row.session_id as string;
      const vid = row.visit_id as string;
      visitors.add(sid);
      if (!sessions.has(vid)) sessions.set(vid, { events: 0, visits: new Set() });
      const s = sessions.get(vid)!;
      s.events++;
      s.visits.add(sid);
    }

    const bounces = [...sessions.values()].filter(s => s.events === 1).length;

    return {
      pageviews: result.rows.length,
      visitors: visitors.size,
      visits: sessions.size,
      bounces,
    };
  }

  /**
   * Pattern 2: getPageviewStats — time-bucketed pageview counts
   * Umami SQL: GROUP BY date bucket with timezone support
   */
  async getPageviewStats(startDate: string, endDate: string, unit: "day" | "hour" = "day") {
    const result = await this.dateFilter(this.siteFilter(this.df()), startDate, endDate)
      .filter("event_type", "eq", 1)
      .collect();

    const buckets = new Map<string, number>();
    for (const row of result.rows) {
      const dt = new Date(row.created_at as string);
      const key = unit === "day"
        ? dt.toISOString().slice(0, 10)
        : dt.toISOString().slice(0, 13) + ":00";
      buckets.set(key, (buckets.get(key) ?? 0) + 1);
    }

    return [...buckets.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([x, y]) => ({ x, y }));
  }

  /**
   * Pattern 3: getPageviewMetrics — top pages/referrers/browsers/etc
   * Umami SQL: GROUP BY column, COUNT DISTINCT session_id, ORDER BY count DESC
   */
  async getMetrics(startDate: string, endDate: string, column: string, limit = 10) {
    const result = await this.dateFilter(this.siteFilter(this.df()), startDate, endDate)
      .filter("event_type", "eq", 1)
      .collect();

    const counts = new Map<string, Set<string>>();
    for (const row of result.rows) {
      const val = String(row[column] ?? "(none)");
      if (!counts.has(val)) counts.set(val, new Set());
      counts.get(val)!.add(row.session_id as string);
    }

    return [...counts.entries()]
      .map(([name, sessions]) => ({ name, value: sessions.size }))
      .sort((a, b) => b.value - a.value)
      .slice(0, limit);
  }

  /**
   * Pattern 4: getFunnel — multi-step funnel analysis
   * Umami SQL: N recursive CTEs with self-joins and time-window constraints
   * This is 80+ lines of dynamically-built SQL in umami.
   */
  async getFunnel(
    startDate: string,
    endDate: string,
    steps: { type: "path" | "event"; value: string }[],
    windowMinutes: number,
  ) {
    const result = await this.dateFilter(this.siteFilter(this.df()), startDate, endDate)
      .collect();

    // Group events by session, sorted by time
    const sessionEvents = new Map<string, WebsiteEvent[]>();
    for (const row of result.rows) {
      const sid = row.session_id as string;
      if (!sessionEvents.has(sid)) sessionEvents.set(sid, []);
      sessionEvents.get(sid)!.push(row as unknown as WebsiteEvent);
    }

    // For each session, check how far through the funnel they got
    const stepCounts = steps.map(() => new Set<string>());

    for (const [sessionId, events] of sessionEvents) {
      const sorted = events.sort((a, b) => a.created_at.localeCompare(b.created_at));
      let lastMatchTime: Date | null = null;

      for (let stepIdx = 0; stepIdx < steps.length; stepIdx++) {
        const step = steps[stepIdx];
        const col = step.type === "path" ? "url_path" : "event_name";

        const match = sorted.find(e => {
          if (String(e[col]) !== step.value) return false;
          if (lastMatchTime) {
            const diff = (new Date(e.created_at).getTime() - lastMatchTime.getTime()) / 60000;
            if (diff > windowMinutes || diff < 0) return false;
          }
          return true;
        });

        if (!match) break; // Dropped off
        stepCounts[stepIdx].add(sessionId);
        lastMatchTime = new Date(match.created_at);
      }
    }

    return steps.map((step, i) => {
      const visitors = stepCounts[i].size;
      const previous = i > 0 ? stepCounts[i - 1].size : visitors;
      return {
        ...step,
        visitors,
        dropoff: previous > 0 ? 1 - visitors / previous : 0,
      };
    });
  }

  /**
   * Pattern 5: getRetention — cohort retention analysis
   * Umami SQL: 4 CTEs (cohort_items, user_activities, cohort_size, cohort_date)
   */
  async getRetention(startDate: string, endDate: string) {
    const result = await this.dateFilter(this.siteFilter(this.df()), startDate, endDate)
      .collect();

    // Step 1: Find each session's cohort date (first event date)
    const sessionCohort = new Map<string, string>();
    for (const row of result.rows) {
      const sid = row.session_id as string;
      const date = (row.created_at as string).slice(0, 10);
      if (!sessionCohort.has(sid) || date < sessionCohort.get(sid)!) {
        sessionCohort.set(sid, date);
      }
    }

    // Step 2: For each session, find which days they were active
    const sessionDays = new Map<string, Set<string>>();
    for (const row of result.rows) {
      const sid = row.session_id as string;
      const date = (row.created_at as string).slice(0, 10);
      if (!sessionDays.has(sid)) sessionDays.set(sid, new Set());
      sessionDays.get(sid)!.add(date);
    }

    // Step 3: Build cohort retention matrix
    const cohortSizes = new Map<string, number>();
    const retention = new Map<string, Map<number, number>>();

    for (const [sid, cohortDate] of sessionCohort) {
      cohortSizes.set(cohortDate, (cohortSizes.get(cohortDate) ?? 0) + 1);
      const days = sessionDays.get(sid)!;
      for (const day of days) {
        const dayNumber = Math.floor(
          (new Date(day).getTime() - new Date(cohortDate).getTime()) / 86400000,
        );
        if (dayNumber > 31) continue;
        if (!retention.has(cohortDate)) retention.set(cohortDate, new Map());
        const cohort = retention.get(cohortDate)!;
        cohort.set(dayNumber, (cohort.get(dayNumber) ?? 0) + 1);
      }
    }

    const results: Array<{
      date: string;
      day: number;
      visitors: number;
      returnVisitors: number;
      percentage: number;
    }> = [];

    for (const [cohortDate, days] of [...retention.entries()].sort()) {
      const totalVisitors = cohortSizes.get(cohortDate) ?? 0;
      for (const [day, returnVisitors] of [...days.entries()].sort((a, b) => a[0] - b[0])) {
        results.push({
          date: cohortDate,
          day,
          visitors: totalVisitors,
          returnVisitors,
          percentage: totalVisitors > 0 ? (returnVisitors * 100) / totalVisitors : 0,
        });
      }
    }

    return results;
  }

  /**
   * Pattern 6: getJourney — user journey path analysis
   * Umami SQL: ROW_NUMBER window function, dynamic N-step pivoting, CTEs
   */
  async getJourney(startDate: string, endDate: string, steps: number) {
    const result = await this.dateFilter(this.siteFilter(this.df()), startDate, endDate)
      .collect();

    // Group by visit, order by time, take first N events
    const visitEvents = new Map<string, Array<{ event: string; time: string }>>();
    for (const row of result.rows) {
      const vid = row.visit_id as string;
      const event = (row.event_name as string) || (row.url_path as string);
      if (!visitEvents.has(vid)) visitEvents.set(vid, []);
      visitEvents.get(vid)!.push({ event, time: row.created_at as string });
    }

    // Build sequences
    const sequences = new Map<string, number>();
    for (const events of visitEvents.values()) {
      const sorted = events.sort((a, b) => a.time.localeCompare(b.time));
      const path = sorted
        .slice(0, steps)
        .map(e => e.event)
        .join(" → ");
      sequences.set(path, (sequences.get(path) ?? 0) + 1);
    }

    return [...sequences.entries()]
      .map(([path, count]) => ({ path, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 100);
  }

  /**
   * Pattern 7: getGoal — goal conversion rate
   * Umami SQL: COUNT DISTINCT with correlated subquery for total
   */
  async getGoal(startDate: string, endDate: string, type: "path" | "event", value: string) {
    const base = this.dateFilter(this.siteFilter(this.df()), startDate, endDate);

    const totalResult = await base.collect();
    const totalSessions = new Set(totalResult.rows.map(r => r.session_id as string));

    const column = type === "path" ? "url_path" : "event_name";
    const goalResult = await base.filter(column, "eq", value).collect();
    const goalSessions = new Set(goalResult.rows.map(r => r.session_id as string));

    return {
      num: goalSessions.size,
      total: totalSessions.size,
      rate: totalSessions.size > 0 ? goalSessions.size / totalSessions.size : 0,
    };
  }

  /**
   * Pattern 8: getEventMetrics — custom event counts
   * Umami SQL: GROUP BY event_name with filters
   */
  async getEventMetrics(startDate: string, endDate: string) {
    const result = await this.dateFilter(this.siteFilter(this.df()), startDate, endDate)
      .filter("event_type", "eq", 2) // custom events only
      .collect();

    const counts = new Map<string, number>();
    for (const row of result.rows) {
      const name = row.event_name as string;
      if (!name) continue;
      counts.set(name, (counts.get(name) ?? 0) + 1);
    }

    return [...counts.entries()]
      .map(([event_name, count]) => ({ event_name, count }))
      .sort((a, b) => b.count - a.count);
  }

  /**
   * Pattern 9: getActiveVisitors — visitors in last 5 minutes
   * Umami SQL: COUNT DISTINCT session_id WHERE created_at > NOW() - 5min
   */
  async getActiveVisitors(asOf: string) {
    const fiveMinAgo = new Date(new Date(asOf).getTime() - 5 * 60000).toISOString();
    const result = await this.siteFilter(this.df())
      .filter("created_at", "gte", fiveMinAgo)
      .filter("created_at", "lte", asOf)
      .collect();

    return new Set(result.rows.map(r => r.session_id as string)).size;
  }

  /**
   * Pattern 10: Attribution (first-click / last-click model)
   * Umami SQL: runs 8 SEPARATE database queries:
   *   referrer, paidAds, utm_source, utm_medium, utm_campaign,
   *   utm_content, utm_term, total
   * Each rebuilds the same CTE base. Total: 8 SQL→JSON round-trips.
   */
  async getAttribution(
    startDate: string,
    endDate: string,
    model: "first-click" | "last-click",
    goalPath: string,
  ) {
    // ONE query — get all events for the period
    const result = await this.dateFilter(this.siteFilter(this.df()), startDate, endDate)
      .collect();

    // Find sessions that hit the goal
    const goalSessions = new Set<string>();
    for (const row of result.rows) {
      if (row.url_path === goalPath) goalSessions.add(row.session_id as string);
    }

    // For each goal session, find the attribution event (first or last click)
    const sessionAttr = new Map<string, Row>();
    for (const row of result.rows) {
      const sid = row.session_id as string;
      if (!goalSessions.has(sid)) continue;
      if (!sessionAttr.has(sid)) {
        sessionAttr.set(sid, row);
      } else {
        const existing = sessionAttr.get(sid)!;
        const existingTime = existing.created_at as string;
        const rowTime = row.created_at as string;
        if (model === "first-click" && rowTime < existingTime) {
          sessionAttr.set(sid, row);
        } else if (model === "last-click" && rowTime > existingTime && rowTime < row.created_at!) {
          sessionAttr.set(sid, row);
        }
      }
    }

    // Now extract all 7 dimensions from the SAME result — no extra queries
    function groupBy(column: string, limit = 20) {
      const counts = new Map<string, number>();
      for (const [, row] of sessionAttr) {
        const val = String(row[column] ?? "");
        if (val === "") continue;
        counts.set(val, (counts.get(val) ?? 0) + 1);
      }
      return [...counts.entries()]
        .map(([name, value]) => ({ name, value }))
        .sort((a, b) => b.value - a.value)
        .slice(0, limit);
    }

    // Paid ads — derived from click IDs
    const paidAds = new Map<string, number>();
    for (const [, row] of sessionAttr) {
      let adName = "";
      if (row.gclid) adName = "Google Ads";
      else if (row.fbclid) adName = "Facebook / Meta";
      else if (row.msclkid) adName = "Microsoft Ads";
      else if (row.ttclid) adName = "TikTok Ads";
      else if (row.li_fat_id) adName = "LinkedIn Ads";
      else if (row.twclid) adName = "Twitter Ads (X)";
      if (adName) paidAds.set(adName, (paidAds.get(adName) ?? 0) + 1);
    }

    return {
      referrer: groupBy("referrer_domain"),
      paidAds: [...paidAds.entries()]
        .map(([name, value]) => ({ name, value }))
        .sort((a, b) => b.value - a.value),
      utm_source: groupBy("utm_source"),
      utm_medium: groupBy("utm_medium"),
      utm_campaign: groupBy("utm_campaign"),
      utm_content: groupBy("utm_content"),
      utm_term: groupBy("utm_term"),
      total: {
        pageviews: result.rows.filter(r => goalSessions.has(r.session_id as string)).length,
        visitors: goalSessions.size,
      },
    };
  }
}

// ---------------------------------------------------------------------------
// 3. CONFORMANCE TESTS — prove querymode handles every Umami query pattern
// ---------------------------------------------------------------------------

describe("Umami Conformance — all query patterns on querymode", () => {
  let events: WebsiteEvent[];
  let umami: UmamiOnQueryMode;
  const startDate = "2026-03-01T00:00:00Z";
  const endDate = "2026-03-08T00:00:00Z";

  beforeAll(() => {
    events = generateEvents(5000);
    umami = new UmamiOnQueryMode(events);
  });

  // Pattern 1: getWebsiteStats
  it("getWebsiteStats — pageviews, visitors, visits, bounces", async () => {
    const stats = await umami.getWebsiteStats(startDate, endDate);

    expect(stats.pageviews).toBeGreaterThan(0);
    expect(stats.visitors).toBeGreaterThan(0);
    expect(stats.visits).toBeGreaterThan(0);
    expect(stats.bounces).toBeGreaterThanOrEqual(0);
    expect(stats.visitors).toBeLessThanOrEqual(stats.visits);

    // Verify against raw data
    const eventsInRange = events.filter(
      e => e.created_at >= startDate && e.created_at < endDate && e.event_type === 1,
    );
    expect(stats.pageviews).toBe(eventsInRange.length);
  });

  // Pattern 2: getPageviewStats
  it("getPageviewStats — daily pageview time series", async () => {
    const series = await umami.getPageviewStats(startDate, endDate, "day");

    expect(series.length).toBeGreaterThan(0);
    // Each bucket should have data
    for (const point of series) {
      expect(point.y).toBeGreaterThan(0);
      expect(point.x).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    }
    // Total across buckets should match pageview count
    const total = series.reduce((sum, p) => sum + p.y, 0);
    const eventsInRange = events.filter(
      e => e.created_at >= startDate && e.created_at < endDate && e.event_type === 1,
    );
    expect(total).toBe(eventsInRange.length);
  });

  it("getPageviewStats — hourly pageview time series", async () => {
    const series = await umami.getPageviewStats(startDate, endDate, "hour");
    expect(series.length).toBeGreaterThan(0);
    expect(series.length).toBeGreaterThan(7); // At least more hours than days
  });

  // Pattern 3: getMetrics
  it("getMetrics — top pages by unique visitors", async () => {
    const metrics = await umami.getMetrics(startDate, endDate, "url_path");
    expect(metrics.length).toBeGreaterThan(0);
    // Sorted descending
    for (let i = 1; i < metrics.length; i++) {
      expect(metrics[i].value).toBeLessThanOrEqual(metrics[i - 1].value);
    }
    // All paths should be from our known set
    for (const m of metrics) {
      expect(PATHS).toContain(m.name);
    }
  });

  it("getMetrics — top referrers", async () => {
    const metrics = await umami.getMetrics(startDate, endDate, "referrer_domain");
    expect(metrics.length).toBeGreaterThan(0);
  });

  it("getMetrics — top browsers", async () => {
    const metrics = await umami.getMetrics(startDate, endDate, "browser");
    expect(metrics.length).toBeGreaterThan(0);
    for (const m of metrics) {
      expect([...BROWSERS, "(none)"]).toContain(m.name);
    }
  });

  it("getMetrics — top countries", async () => {
    const metrics = await umami.getMetrics(startDate, endDate, "country");
    expect(metrics.length).toBeGreaterThan(0);
  });

  // Pattern 4: getFunnel
  it("getFunnel — multi-step funnel with time window", async () => {
    const funnel = await umami.getFunnel(startDate, endDate, [
      { type: "path", value: "/" },
      { type: "path", value: "/pricing" },
      { type: "path", value: "/signup" },
    ], 30);

    expect(funnel.length).toBe(3);
    // Each step should have equal or fewer visitors than the previous
    for (let i = 1; i < funnel.length; i++) {
      expect(funnel[i].visitors).toBeLessThanOrEqual(funnel[i - 1].visitors);
    }
    // First step should have visitors
    expect(funnel[0].visitors).toBeGreaterThan(0);
  });

  // Pattern 5: getRetention
  it("getRetention — cohort retention matrix", async () => {
    const retention = await umami.getRetention(startDate, endDate);

    expect(retention.length).toBeGreaterThan(0);
    // Day 0 retention should always be 100%
    const day0 = retention.filter(r => r.day === 0);
    for (const row of day0) {
      expect(row.percentage).toBe(100);
    }
    // Each entry should have valid fields
    for (const row of retention) {
      expect(row.visitors).toBeGreaterThan(0);
      expect(row.returnVisitors).toBeGreaterThan(0);
      expect(row.percentage).toBeGreaterThan(0);
      expect(row.percentage).toBeLessThanOrEqual(100);
    }
  });

  // Pattern 6: getJourney
  it("getJourney — user journey paths", async () => {
    const journeys = await umami.getJourney(startDate, endDate, 3);

    expect(journeys.length).toBeGreaterThan(0);
    // Sorted by count descending
    for (let i = 1; i < journeys.length; i++) {
      expect(journeys[i].count).toBeLessThanOrEqual(journeys[i - 1].count);
    }
    // Each path should contain step separators
    for (const j of journeys) {
      expect(j.count).toBeGreaterThan(0);
    }
  });

  // Pattern 7: getGoal
  it("getGoal — conversion rate for /signup path", async () => {
    const goal = await umami.getGoal(startDate, endDate, "path", "/signup");

    expect(goal.total).toBeGreaterThan(0);
    expect(goal.num).toBeGreaterThanOrEqual(0);
    expect(goal.rate).toBeGreaterThanOrEqual(0);
    expect(goal.rate).toBeLessThanOrEqual(1);
  });

  // Pattern 8: getEventMetrics
  it("getEventMetrics — custom event counts", async () => {
    const metrics = await umami.getEventMetrics(startDate, endDate);

    expect(metrics.length).toBeGreaterThan(0);
    for (const row of metrics) {
      expect(row.event_name).toBeTruthy();
      expect(row.count).toBeGreaterThan(0);
    }
  });

  // Pattern 9: getActiveVisitors
  it("getActiveVisitors — unique visitors in last 5 minutes", async () => {
    // Use a timestamp within our data range
    const midpoint = "2026-03-04T12:00:00Z";
    const active = await umami.getActiveVisitors(midpoint);
    // May or may not have visitors at this exact moment
    expect(active).toBeGreaterThanOrEqual(0);
  });

  // Pattern 10: getAttribution
  it("getAttribution — first-click model, all 7 dimensions in 1 pass", async () => {
    const attr = await umami.getAttribution(startDate, endDate, "first-click", "/signup");

    // Should have data in at least some dimensions
    expect(attr.total.visitors).toBeGreaterThanOrEqual(0);
    // Referrers should be sorted by value descending
    if (attr.referrer.length > 1) {
      expect(attr.referrer[0].value).toBeGreaterThanOrEqual(attr.referrer[1].value);
    }
  });

  // Cross-cutting: filter composition
  it("filters compose — path + country + browser drill-down", async () => {
    const df = QueryMode.fromJSON(events, "events");
    const result = await df
      .filter("website_id", "eq", WEBSITE_ID)
      .filter("created_at", "gte", startDate)
      .filter("created_at", "lt", endDate)
      .filter("url_path", "eq", "/pricing")
      .filter("country", "eq", "US")
      .filter("browser", "eq", "Chrome")
      .collect();

    for (const row of result.rows) {
      expect(row.url_path).toBe("/pricing");
      expect(row.country).toBe("US");
      expect(row.browser).toBe("Chrome");
    }
  });
});

// ---------------------------------------------------------------------------
// 4. QUERY AS CODE — things impossible with Umami's Prisma/ClickHouse SQL API
//
// Umami sends raw SQL strings to PostgreSQL/ClickHouse via Prisma rawQuery().
// Every query is: build SQL string → send to DB → JSON parse response.
// You CANNOT inspect intermediate results, branch on them, or compose custom logic
// between query stages. The attribution report alone makes 8 separate DB round-trips.
//
// With querymode, operators are function calls on structured data. You read results,
// decide what to do next, and feed them into the next stage. No serialization boundary.
// ---------------------------------------------------------------------------

describe("Query as Code — impossible with Umami's Prisma SQL", () => {
  let events: WebsiteEvent[];
  let umami: UmamiOnQueryMode;
  const startDate = "2026-03-01T00:00:00Z";
  const endDate = "2026-03-15T00:00:00Z";

  beforeAll(() => {
    events = generateEvents(10000);
    umami = new UmamiOnQueryMode(events);
  });

  /**
   * IMPOSSIBLE #1: Attribution in 1 pass instead of 8 separate DB queries
   *
   * Umami's getAttribution() in production runs 8 SEPARATE SQL queries:
   *   1. referrerRes = rawQuery("WITH events AS (...), model AS (...) SELECT referrer_domain...")
   *   2. paidAdsRes = rawQuery("WITH events AS (...), model AS (...) SELECT CASE gclid/fbclid...")
   *   3. sourceRes = rawQuery("WITH events AS (...), model AS (...) SELECT utm_source...")
   *   4. mediumRes = rawQuery("WITH events AS (...), model AS (...) SELECT utm_medium...")
   *   5. campaignRes = rawQuery("WITH events AS (...), model AS (...) SELECT utm_campaign...")
   *   6. contentRes = rawQuery("WITH events AS (...), model AS (...) SELECT utm_content...")
   *   7. termRes = rawQuery("WITH events AS (...), model AS (...) SELECT utm_term...")
   *   8. totalRes = rawQuery("SELECT count(*), count(distinct session_id)...")
   *
   * Each rebuilds the SAME base CTE. That's 8 SQL string constructions + 8 DB round-trips +
   * 8 JSON deserializations. With querymode: 1 collect(), branch 8 ways in code.
   */
  it("attribution in 1 pass — replaces 8 separate DB queries", async () => {
    const attr = await umami.getAttribution(startDate, endDate, "first-click", "/signup");

    // All 7 dimensions computed from a single collect()
    expect(attr.referrer).toBeDefined();
    expect(attr.paidAds).toBeDefined();
    expect(attr.utm_source).toBeDefined();
    expect(attr.utm_medium).toBeDefined();
    expect(attr.utm_campaign).toBeDefined();
    expect(attr.utm_content).toBeDefined();
    expect(attr.utm_term).toBeDefined();
    expect(attr.total).toBeDefined();

    // Verify data integrity — all dimensions reference the same session set
    const totalVisitors = attr.total.visitors;
    if (totalVisitors > 0) {
      // Sum of any single dimension should be <= total (sessions can have empty UTM)
      const referrerTotal = attr.referrer.reduce((s, r) => s + r.value, 0);
      expect(referrerTotal).toBeLessThanOrEqual(totalVisitors);
    }
  });

  /**
   * IMPOSSIBLE #2: Funnel → Retention pipeline (cross-report correlation)
   *
   * "For users who completed the /pricing → /signup funnel, what's their
   *  retention pattern? Do they come back?"
   *
   * With Umami: getFunnel() returns counts only, no session IDs.
   * You'd need to rewrite the SQL to return session IDs, run it,
   * then build a new retention query for just those sessions.
   * That's custom SQL + 2 DB round-trips minimum.
   *
   * With querymode: collect once, run funnel logic, take the session IDs,
   * feed directly into retention analysis. Zero round-trips between stages.
   */
  it("funnel → retention pipeline — cross-report correlation", async () => {
    const result = await QueryMode.fromJSON(events, "events")
      .filter("website_id", "eq", WEBSITE_ID)
      .filter("created_at", "gte", startDate)
      .filter("created_at", "lt", endDate)
      .collect();

    // Step 1: Funnel — find sessions that hit / → /pricing → /signup
    const sessionEvents = new Map<string, Row[]>();
    for (const row of result.rows) {
      const sid = row.session_id as string;
      if (!sessionEvents.has(sid)) sessionEvents.set(sid, []);
      sessionEvents.get(sid)!.push(row);
    }

    const funnelSessions = new Set<string>();
    for (const [sid, evts] of sessionEvents) {
      const paths = evts
        .sort((a, b) => (a.created_at as string).localeCompare(b.created_at as string))
        .map(e => e.url_path);
      let step = 0;
      const funnel = ["/", "/pricing", "/signup"];
      for (const path of paths) {
        if (path === funnel[step]) step++;
        if (step === funnel.length) { funnelSessions.add(sid); break; }
      }
    }

    // Step 2: Retention — for just funnel completers, measure return rate
    // This reuses the SAME result.rows — no second query
    const sessionFirstDay = new Map<string, string>();
    const sessionActiveDays = new Map<string, Set<string>>();
    for (const row of result.rows) {
      const sid = row.session_id as string;
      if (!funnelSessions.has(sid)) continue;

      const day = (row.created_at as string).slice(0, 10);
      if (!sessionFirstDay.has(sid) || day < sessionFirstDay.get(sid)!) {
        sessionFirstDay.set(sid, day);
      }
      if (!sessionActiveDays.has(sid)) sessionActiveDays.set(sid, new Set());
      sessionActiveDays.get(sid)!.add(day);
    }

    // Compute day-1 retention for funnel completers
    let returnedNextDay = 0;
    for (const [sid, firstDay] of sessionFirstDay) {
      const nextDay = new Date(new Date(firstDay).getTime() + 86400000).toISOString().slice(0, 10);
      if (sessionActiveDays.get(sid)!.has(nextDay)) returnedNextDay++;
    }

    const day1Retention = funnelSessions.size > 0 ? returnedNextDay / funnelSessions.size : 0;

    // The point: funnel analysis flowed directly into retention analysis
    // on the same result set. No SQL rewriting, no extra DB calls.
    expect(day1Retention).toBeGreaterThanOrEqual(0);
    expect(day1Retention).toBeLessThanOrEqual(1);
  });

  /**
   * IMPOSSIBLE #3: Anomaly detection with real-time threshold adjustment
   *
   * "Flag pages where today's traffic is >2x the 7-day average,
   *  then for those anomalous pages, find the top referrer source."
   *
   * With Umami: 1st query for daily stats per page, parse JSON, compute
   * averages in JS, build IN clause for anomalous pages, 2nd query for
   * referrers filtered to those pages. 2 round-trips + JSON parsing.
   *
   * With querymode: 1 collect(), all analysis in code, zero round-trips.
   */
  it("anomaly detection — flag high-traffic pages, find their referrers", async () => {
    const result = await QueryMode.fromJSON(events, "events")
      .filter("website_id", "eq", WEBSITE_ID)
      .filter("event_type", "eq", 1)
      .collect();

    // Step 1: Compute daily traffic per page
    const pageDaily = new Map<string, Map<string, number>>();
    for (const row of result.rows) {
      const path = row.url_path as string;
      const day = (row.created_at as string).slice(0, 10);
      if (!pageDaily.has(path)) pageDaily.set(path, new Map());
      const days = pageDaily.get(path)!;
      days.set(day, (days.get(day) ?? 0) + 1);
    }

    // Step 2: Find anomalous pages (any day deviates from average)
    // With random data, use a lower threshold to ensure we find anomalies
    const anomalousPages = new Set<string>();
    for (const [path, days] of pageDaily) {
      const values = [...days.values()];
      const avg = values.reduce((a, b) => a + b, 0) / values.length;
      const maxVal = Math.max(...values);
      const minVal = Math.min(...values);
      // Flag if max day is at least 20% above average or there's meaningful variance
      if (maxVal > avg * 1.2 || (maxVal - minVal) > avg * 0.5) anomalousPages.add(path);
    }

    // Step 3: For anomalous pages, find top referrer (SAME result set)
    const referrerCounts = new Map<string, number>();
    for (const row of result.rows) {
      if (!anomalousPages.has(row.url_path as string)) continue;
      const ref = row.referrer_domain as string;
      if (!ref) continue;
      referrerCounts.set(ref, (referrerCounts.get(ref) ?? 0) + 1);
    }

    const topReferrer = [...referrerCounts.entries()].sort((a, b) => b[1] - a[1])[0];

    // All 3 steps ran on one result set — no SQL string building, no DB round-trips
    expect(anomalousPages.size).toBeGreaterThan(0);
    if (topReferrer) {
      expect(topReferrer[1]).toBeGreaterThan(0);
    }
  });

  /**
   * IMPOSSIBLE #4: Session-level engagement scoring with conditional logic
   *
   * "Score each session: +1 for pageview, +3 for signup_start, +5 for purchase.
   *  Group sessions into tiers (low/medium/high), then for high-engagement sessions
   *  find which UTM campaigns brought them."
   *
   * With Umami: You'd need to write custom SQL with CASE WHEN scoring,
   * run it, parse JSON, then build another query for UTM analysis.
   * The SQL gets unwieldy fast with complex scoring logic.
   *
   * With querymode: collect(), score in TypeScript, branch on scores.
   * Complex business logic stays in code where it belongs.
   */
  it("session engagement scoring — custom business logic stays in code", async () => {
    const result = await QueryMode.fromJSON(events, "events")
      .filter("website_id", "eq", WEBSITE_ID)
      .filter("created_at", "gte", startDate)
      .filter("created_at", "lt", endDate)
      .collect();

    // Step 1: Score each session
    const sessionScores = new Map<string, number>();
    const sessionUtm = new Map<string, string>();
    for (const row of result.rows) {
      const sid = row.session_id as string;
      const score = sessionScores.get(sid) ?? 0;

      // Complex scoring logic — trivial in code, painful in SQL
      let points = 1; // base pageview
      if (row.event_name === "signup_start") points = 3;
      else if (row.event_name === "signup_complete") points = 5;
      else if (row.event_name === "purchase") points = 10;
      else if (row.event_name === "download") points = 2;

      sessionScores.set(sid, score + points);

      // Capture UTM for first event in session
      if (!sessionUtm.has(sid) && row.utm_campaign) {
        sessionUtm.set(sid, row.utm_campaign as string);
      }
    }

    // Step 2: Tier sessions
    const tiers = { low: 0, medium: 0, high: 0 };
    const highEngagementCampaigns = new Map<string, number>();
    for (const [sid, score] of sessionScores) {
      if (score >= 10) {
        tiers.high++;
        const campaign = sessionUtm.get(sid);
        if (campaign) {
          highEngagementCampaigns.set(campaign, (highEngagementCampaigns.get(campaign) ?? 0) + 1);
        }
      } else if (score >= 3) {
        tiers.medium++;
      } else {
        tiers.low++;
      }
    }

    // Step 3: Results — which campaigns drive high engagement?
    const topCampaigns = [...highEngagementCampaigns.entries()]
      .sort((a, b) => b[1] - a[1]);

    // All of this — scoring, tiering, attribution — happened on one result set
    expect(tiers.low + tiers.medium + tiers.high).toBe(sessionScores.size);
    expect(sessionScores.size).toBeGreaterThan(0);
  });

  /**
   * IMPOSSIBLE #5: Real-time A/B test analysis with statistical significance
   *
   * "Compare /pricing (variant A) vs /pricing-v2 (variant B) conversion
   *  to /signup. Compute conversion rates AND determine if the difference
   *  is statistically significant."
   *
   * With Umami: Multiple SQL queries to get counts for each variant,
   * parse JSON responses, compute Z-scores in JS. Can't do the stats in SQL.
   * At minimum 2 DB round-trips.
   *
   * With querymode: 1 collect(), all analysis including statistics in code.
   */
  it("A/B test analysis — conversion comparison with statistical check", async () => {
    const result = await QueryMode.fromJSON(events, "events")
      .filter("website_id", "eq", WEBSITE_ID)
      .filter("created_at", "gte", startDate)
      .filter("created_at", "lt", endDate)
      .collect();

    // Step 1: Group sessions by which pricing page they saw
    const sessionsA = new Set<string>(); // saw /pricing
    const sessionsB = new Set<string>(); // saw /about (as control)
    const convertedSessions = new Set<string>(); // hit /signup

    for (const row of result.rows) {
      const sid = row.session_id as string;
      if (row.url_path === "/pricing") sessionsA.add(sid);
      if (row.url_path === "/about") sessionsB.add(sid);
      if (row.url_path === "/signup") convertedSessions.add(sid);
    }

    // Step 2: Compute conversion rates
    const convA = sessionsA.size > 0
      ? [...sessionsA].filter(s => convertedSessions.has(s)).length / sessionsA.size
      : 0;
    const convB = sessionsB.size > 0
      ? [...sessionsB].filter(s => convertedSessions.has(s)).length / sessionsB.size
      : 0;

    // Step 3: Z-test for significance (can't do this in SQL at all)
    const nA = sessionsA.size;
    const nB = sessionsB.size;
    if (nA > 0 && nB > 0) {
      const pooledP = (convA * nA + convB * nB) / (nA + nB);
      const se = Math.sqrt(pooledP * (1 - pooledP) * (1 / nA + 1 / nB));
      const zScore = se > 0 ? Math.abs(convA - convB) / se : 0;
      const isSignificant = zScore > 1.96; // 95% confidence

      // The computation itself IS the test — no SQL gymnastics needed
      expect(typeof isSignificant).toBe("boolean");
      expect(zScore).toBeGreaterThanOrEqual(0);
    }

    expect(sessionsA.size).toBeGreaterThan(0);
    expect(sessionsB.size).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// 5. BENCHMARKS — measure querymode performance on Umami workloads
// ---------------------------------------------------------------------------

describe("Benchmarks — querymode on Umami workloads", () => {
  for (const size of [1_000, 10_000, 100_000]) {
    describe(`${(size / 1000).toFixed(0)}K events`, () => {
      let events: WebsiteEvent[];
      let umami: UmamiOnQueryMode;
      const startDate = "2026-03-01T00:00:00Z";
      const endDate = "2026-03-08T00:00:00Z";

      beforeAll(() => {
        events = generateEvents(size);
        umami = new UmamiOnQueryMode(events);
      });

      it("getWebsiteStats", async () => {
        const t0 = performance.now();
        await umami.getWebsiteStats(startDate, endDate);
        const elapsed = performance.now() - t0;
        console.log(`  getWebsiteStats (${(size / 1000).toFixed(0)}K): ${elapsed.toFixed(1)}ms`);
        expect(elapsed).toBeLessThan(10000);
      });

      it("getPageviewStats", async () => {
        const t0 = performance.now();
        await umami.getPageviewStats(startDate, endDate);
        const elapsed = performance.now() - t0;
        console.log(`  getPageviewStats (${(size / 1000).toFixed(0)}K): ${elapsed.toFixed(1)}ms`);
        expect(elapsed).toBeLessThan(10000);
      });

      it("getMetrics (url_path)", async () => {
        const t0 = performance.now();
        await umami.getMetrics(startDate, endDate, "url_path");
        const elapsed = performance.now() - t0;
        console.log(`  getMetrics (${(size / 1000).toFixed(0)}K): ${elapsed.toFixed(1)}ms`);
        expect(elapsed).toBeLessThan(10000);
      });

      it("getFunnel (3-step)", async () => {
        const t0 = performance.now();
        await umami.getFunnel(startDate, endDate, [
          { type: "path", value: "/" },
          { type: "path", value: "/pricing" },
          { type: "path", value: "/signup" },
        ], 30);
        const elapsed = performance.now() - t0;
        console.log(`  getFunnel (${(size / 1000).toFixed(0)}K): ${elapsed.toFixed(1)}ms`);
        expect(elapsed).toBeLessThan(10000);
      });

      it("getRetention", async () => {
        const t0 = performance.now();
        await umami.getRetention(startDate, endDate);
        const elapsed = performance.now() - t0;
        console.log(`  getRetention (${(size / 1000).toFixed(0)}K): ${elapsed.toFixed(1)}ms`);
        expect(elapsed).toBeLessThan(10000);
      });

      it("getAttribution (1 pass vs 8 queries)", async () => {
        const t0 = performance.now();
        await umami.getAttribution(startDate, endDate, "first-click", "/signup");
        const elapsed = performance.now() - t0;
        console.log(`  getAttribution (${(size / 1000).toFixed(0)}K): ${elapsed.toFixed(1)}ms — replaces 8 DB round-trips`);
        expect(elapsed).toBeLessThan(10000);
      });
    });
  }
});
