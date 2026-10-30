import styles from "./WorkspaceBackdrop.module.css";

/**
 * Drawings behind the study workspace, fixed to the screen rather than the
 * page, so they stay put while the content scrolls over them.
 */
export function WorkspaceBackdrop() {
  return (
    <>
      <div className={styles.topRight} aria-hidden="true" />
      <div className={styles.bottomLeft} aria-hidden="true" />
    </>
  );
}
