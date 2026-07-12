/* Canonical route paths — the single source of truth for react-router `to`
   targets and programmatic navigate() calls, so a path rename is one edit here
   rather than scattered string literals. */
export const ROUTES = {
  identify: "/",
  fieldLog: "/field-log",
  dives: "/dives",
  library: "/library",
  settings: "/settings",
  account: "/account",
  login: "/login",
} as const;
