import styles from "./PortalBackdrop.module.css";

/**
 * Drawn skull behind the portal. Fixed to the screen, not to the page, so it
 * stays put when the content gets shorter or longer (filters, search,
 * paging) and the page scrolls over it.
 */
export function PortalBackdrop() {
  return <div className={styles.backdrop} aria-hidden="true" />;
}
