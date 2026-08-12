import { createServices } from "./services.js";
import { runApp } from "./app.js";

async function main(): Promise<void> {
  const services = await createServices();
  try {
    await runApp(services);
  } finally {
    services.dispose();
  }
}

main().catch((err) => {
  console.error(err instanceof Error ? err.stack || err.message : err);
  process.exitCode = 1;
});
