window.ORBITA_CONFIG = {
  // When served by the Railway app itself, use the same origin as the API.
  // On localhost (dev), default to the production URL with demo mode on.
  defaultApiBase: window.location.hostname.endsWith(".railway.app")
    ? window.location.origin
    : "https://orbita-research-mvp-production.up.railway.app",
  defaultMockMode: !window.location.hostname.endsWith(".railway.app"),
  requestTimeoutMs: 120000,
  routes: {
    cases: "/cases",
    health: "/health"
  }
};
