import styles from "./WorkspaceBackdrop.module.css";

/**
 * Drawing behind the study workspace, fixed to the screen rather than the
 * page, so it stays put while the content scrolls over it.
 */
export function WorkspaceBackdrop() {
  return <div className={styles.bottomLeft} aria-hidden="true" />;
}
