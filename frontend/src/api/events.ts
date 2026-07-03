/* App-wide DOM events — kept in a dependency-free module so both the http layer
   (dispatcher) and AuthContext (listener) can import it without a cycle. */
export const AUTH_EXPIRED_EVENT = "wtf:auth-expired";

/* Backend reachability signal. The http layer (apiFetch) dispatches these so the
   BackendStatus context can reflect cold-start / recovery WITHOUT the fetch layer
   importing React — same decoupling as AUTH_EXPIRED_EVENT. OK = any 2xx observed;
   SLOW = a call is taking long / hit a retryable 5xx / timed out (i.e. the
   instance is probably waking or unreachable). */
export const BACKEND_OK_EVENT = "wtf:backend-ok";
export const BACKEND_SLOW_EVENT = "wtf:backend-slow";
