# Prospecting — Where to Find Interview Candidates

## Segment A: Evaluators (people who've evaluated DuckDB/Polars/DataFusion)

### High-signal sources (start here)

**DuckDB Discord — #showcase and #help**
- People posting projects = active users who can articulate what works/doesn't
- People asking migration questions = in the process of evaluating
- Look for: "I'm trying to do X with DuckDB" or "coming from Pandas/Spark"

**Polars GitHub Discussions**
- "Q&A" category: people asking "can Polars do X?" are evaluating capability gaps
- "Show and tell": people sharing projects = power users who know the limits
- Look for: TypeScript/JS users (rare in Polars community — they're underserved)

**r/dataengineering**
- Search: "duckdb vs" or "polars vs" or "query library"
- Commenters who share their specific setup + pain points
- Sort by recent (last 3 months)

**Hacker News**
- Search: "DuckDB" or "Polars" or "embedded analytics"
- "Show HN" posts about query tools — commenters sharing alternatives are evaluators
- Look for: people who mention Node/TypeScript/Cloudflare/edge

### Medium-signal sources

**Twitter/X**
- Search: "switched from pandas to" OR "duckdb typescript" OR "polars node"
- People tweeting about their stack decisions

**Dev.to / Hashnode / Medium**
- "DuckDB tutorial" or "Polars getting started" authors
- They've done the evaluation — DM and ask about their experience

### Outreach message (adapt per platform)

> Hey — saw your [post/project/comment] about [DuckDB/Polars]. I'm researching how devs evaluate query libraries — what works, what doesn't, where they get stuck. Not selling anything. Would you do a 30-min chat? Happy to share what I learn.

---

## Segment B: Edge/Serverless builders

### High-signal sources

**Cloudflare Discord — #workers, #durable-objects, #d1**
- People asking about data access patterns from Workers
- People hitting D1 limitations and looking for alternatives
- People using R2 for data storage
- Look for: "analytics" or "query" or "aggregate" in worker context

**Vercel Discussions / Next.js Discord**
- People building dashboards or analytics on edge functions
- "How do I query data from edge functions?" type questions
- Look for: frustration with API latency or serverless cold starts

**X/Twitter**
- Search: "duckdb wasm" OR "analytics edge" OR "serverless query" OR "cloudflare workers database"
- People experimenting with embedded query engines in browsers/edge

**GitHub**
- Repos using `@cloudflare/workers-types` + any query library
- Repos with "analytics" + "edge" or "serverless" in description
- Check their issues for data access pain points

### Medium-signal sources

**Deno Discord / Deploy channels**
- Similar to Cloudflare — people running data workloads at edge

**Blog posts**
- Search: "building analytics on cloudflare workers" or "edge function database"
- Authors are practitioners who've hit real limits

### Outreach message

> Hey — saw you're building [data-intensive thing] on [Cloudflare/Vercel/edge]. I'm researching how devs handle analytical queries in serverless/edge environments. 30-min chat? Not pitching — just learning. Happy to share findings.

---

## Prospecting workflow

### Week 1: Source and reach out
- [ ] Spend 2 hours browsing the high-signal sources above
- [ ] Identify 15-20 candidates (mix of A and B segments)
- [ ] Send outreach messages to all 15-20
- [ ] Expect ~30% response rate → 5-6 interviews

### Week 2: Interview
- [ ] Schedule 5-6 interviews (30 min each)
- [ ] Run interviews using research/user-interviews.md guide
- [ ] Fill in research/interview-tracker.md after each
- [ ] Debrief after every 2 interviews — are patterns emerging?

### Week 3: Synthesize and decide
- [ ] Fill in synthesis section of interview-tracker.md
- [ ] Present findings to team
- [ ] Decide: what to build next, what to kill, what to change in messaging
- [ ] Update README if messaging needs to change

### Scheduling tips
- Offer 3 time slots, include one evening/weekend for different timezones
- Use Calendly or cal.com with 30-min slots
- Record with consent (Loom, Zoom recording) for quotes
- Send a thank-you + findings summary after all interviews complete

---

## Tracking

| # | Name/Handle | Source | Segment | Reached out | Responded | Scheduled | Done |
|---|------------|--------|---------|-------------|-----------|-----------|------|
| 1 |            |        |         |             |           |           |      |
| 2 |            |        |         |             |           |           |      |
| 3 |            |        |         |             |           |           |      |
| 4 |            |        |         |             |           |           |      |
| 5 |            |        |         |             |           |           |      |
| 6 |            |        |         |             |           |           |      |
| 7 |            |        |         |             |           |           |      |
| 8 |            |        |         |             |           |           |      |
| 9 |            |        |         |             |           |           |      |
| 10 |           |        |         |             |           |           |      |
