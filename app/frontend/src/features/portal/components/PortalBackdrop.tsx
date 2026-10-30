import styles from "./PortalBackdrop.module.css";

/**
 * Drawn skull and needle holder behind the portal. They are fixed to the
 * screen, not to the page, so they stay put when the content gets shorter or
 * longer (filters, search, paging) and the page scrolls over them.
 */
export function PortalBackdrop() {
  return (
    <>
      <div className={styles.backdrop} aria-hidden="true" />
      <div className={styles.instrument} aria-hidden="true" />
    </>
  );
}
