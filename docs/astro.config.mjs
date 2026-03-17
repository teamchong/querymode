import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

export default defineConfig({
  site: "https://teamchong.github.io",
  base: "/querymode",
  integrations: [
    starlight({
      title: "querymode",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/teamchong/querymode",
        },
      ],
      sidebar: [
        { label: "Overview", slug: "index" },
        { label: "Why QueryMode", slug: "why-querymode" },
        { label: "Getting Started", slug: "getting-started" },
        { label: "DataFrame API", slug: "dataframe-api" },
        { label: "SQL", slug: "sql" },
        { label: "Schema Discovery", slug: "schema" },
        { label: "Operators", slug: "operators" },
        { label: "Composability", slug: "composability" },
        { label: "Pipeline", slug: "pipeline" },
        { label: "Formats", slug: "formats" },
        { label: "Architecture", slug: "architecture" },
        { label: "Columnar Format", slug: "columnar-format" },
        { label: "Lazy Evaluation", slug: "lazy-evaluation" },
        { label: "Performance", slug: "performance" },
        { label: "Write Path", slug: "write-path" },
        { label: "Deployment", slug: "deployment" },
      ],
    }),
  ],
});
