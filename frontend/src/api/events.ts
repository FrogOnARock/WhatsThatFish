/* App-wide DOM events — kept in a dependency-free module so both the http layer
   (dispatcher) and AuthContext (listener) can import it without a cycle. */
export const AUTH_EXPIRED_EVENT = "wtf:auth-expired";
