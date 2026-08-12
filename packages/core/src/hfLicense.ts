/**
 * Hugging Face model-card license helpers for the download picker.
 * Tags (`license:apache-2.0`) come from search; detail fields from /api/models/{id}.
 */

export type LicenseBucket = "permissive" | "limited" | "custom" | "unknown";

export interface ModelLicenseInfo {
  /** SPDX / HF id, e.g. apache-2.0, other, llama3.1, gemma */
  id: string;
  /** Custom name when id is "other" (e.g. lfm1.0). */
  name?: string;
  /** Absolute URL or HF-relative LICENSE path. */
  link?: string;
  bucket: LicenseBucket;
  /** Short badge for QuickPick description. */
  badge: string;
  /** Longer line for confirm dialogs. */
  summary: string;
  /** Codicon without $(), e.g. "warning". */
  icon: string;
  /** True when we should confirm before download. */
  needsConfirm: boolean;
}

interface HfCardData {
  license?: string;
  license_name?: string;
  license_link?: string;
}

interface HfModelDetail {
  id?: string;
  tags?: string[];
  cardData?: HfCardData;
}

const PERMISSIVE = new Set([
  "mit",
  "apache-2.0",
  "apache-2.0-or-later",
  "bsd-2-clause",
  "bsd-3-clause",
  "bsd",
  "cc0-1.0",
  "unlicense",
  "isc",
  "zlib",
  "mpl-2.0",
  "artistic-2.0",
  "0bsd",
]);

function tagLicense(tags: string[] | undefined): string | undefined {
  for (const t of tags || []) {
    const m = /^license:(.+)$/i.exec(t.trim());
    if (m?.[1]) {
      return m[1].toLowerCase();
    }
  }
  return undefined;
}

function normalizeId(raw: string | undefined): string {
  return (raw || "").trim().toLowerCase();
}

function isLfm(id: string, name?: string): boolean {
  const n = (name || "").toLowerCase();
  return id === "lfm1.0" || n.startsWith("lfm") || /^lfm\d/.test(n);
}

function isLlama(id: string, name?: string): boolean {
  const n = (name || "").toLowerCase();
  return id.startsWith("llama") || n.includes("llama");
}

function isGemma(id: string, name?: string): boolean {
  const n = (name || "").toLowerCase();
  return id === "gemma" || n.includes("gemma");
}

/**
 * Classify a license id (+ optional custom name) into a UI bucket.
 */
export function classifyLicense(idRaw: string | undefined, nameRaw?: string): ModelLicenseInfo {
  const id = normalizeId(idRaw) || "unknown";
  const name = nameRaw?.trim() || undefined;

  if (!idRaw && !name) {
    return {
      id: "unknown",
      bucket: "unknown",
      badge: "License unknown",
      summary: "No license metadata on the Hugging Face model card.",
      icon: "question",
      needsConfirm: true,
    };
  }

  if (PERMISSIVE.has(id)) {
    const label = id.toUpperCase() === id ? id : id;
    return {
      id,
      name,
      bucket: "permissive",
      badge: label,
      summary: `Permissive license (${label}).`,
      icon: "pass-filled",
      needsConfirm: false,
    };
  }

  if (isLfm(id, name) || (id === "other" && isLfm(id, name))) {
    const label = name || "LFM 1.0";
    return {
      id: id === "other" ? "other" : id,
      name: label,
      bucket: "limited",
      badge: `${label} · commercial <$10M`,
      summary:
        `${label}: free commercial use only if annual revenue is under $10M USD; ` +
        `above that a paid Liquid AI license is required.`,
      icon: "warning",
      needsConfirm: true,
    };
  }

  if (isLlama(id, name)) {
    const label = name || id;
    return {
      id,
      name,
      bucket: "limited",
      badge: `${label} · commercial <700M MAU`,
      summary:
        `${label}: commercial use allowed for most products, but services with >700M monthly ` +
        `active users need a separate Meta license. Attribution (“Built with Llama”) and the ` +
        `Acceptable Use Policy apply.`,
      icon: "warning",
      needsConfirm: true,
    };
  }

  if (isGemma(id, name)) {
    const label = name || "Gemma Terms";
    return {
      id,
      name,
      bucket: "custom",
      badge: label,
      summary:
        `${label}: commercial use allowed under Google’s Gemma Terms, including a prohibited-use ` +
        `policy that must be passed through to downstream users. (Gemma 4 weights may be Apache-2.0 — check the card.)`,
      icon: "law",
      needsConfirm: true,
    };
  }

  if (id === "other") {
    const label = name || "other";
    return {
      id: "other",
      name,
      bucket: "custom",
      badge: label,
      summary: `Custom license (${label}). Review the model card before commercial use.`,
      icon: "law",
      needsConfirm: true,
    };
  }

  // Other known custom ids (openai, bigscience-openrail-m, …)
  return {
    id,
    name,
    bucket: "custom",
    badge: name || id,
    summary: `Non-standard license (${name || id}). Review the model card before commercial use.`,
    icon: "law",
    needsConfirm: true,
  };
}

/** Fast path from search-result tags only. */
export function licenseFromTags(tags: string[] | undefined): ModelLicenseInfo {
  return classifyLicense(tagLicense(tags));
}

/** Prefer cardData; fall back to tags. */
export function licenseFromModelDetail(detail: HfModelDetail): ModelLicenseInfo {
  const card = detail.cardData || {};
  const id = normalizeId(card.license) || tagLicense(detail.tags);
  const info = classifyLicense(id, card.license_name);
  if (card.license_link) {
    info.link = card.license_link;
  }
  return info;
}

/** Absolute URL for “View license”, or undefined. */
export function resolveLicenseUrl(modelId: string, info: ModelLicenseInfo): string | undefined {
  const link = (info.link || "").trim();
  if (!link) {
    return `https://huggingface.co/${modelId}`;
  }
  if (/^https?:\/\//i.test(link)) {
    return link;
  }
  // Relative path in the repo (e.g. LICENSE)
  const pathPart = link.replace(/^\.\//, "").replace(/^\//, "");
  return `https://huggingface.co/${modelId}/blob/main/${pathPart}`;
}

export function formatLicenseQuickPick(
  modelId: string,
  downloads: number | undefined,
  info: ModelLicenseInfo
): { label: string; description: string } {
  const dl =
    typeof downloads === "number"
      ? `${downloads.toLocaleString()} downloads`
      : "downloads —";
  return {
    label: `$(${info.icon}) ${modelId}`,
    description: `${info.badge} · ${dl}`,
  };
}
