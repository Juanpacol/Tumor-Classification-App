import type { ReactNode } from "react";
import { PortalHeader } from "./PortalHeader";
import styles from "./PortalShell.module.css";

/** Frame of every portal page: the header with the disclaimer, then the page. */
export function PortalShell({ children }: { children: ReactNode }) {
  return (
    <div className="portal">
      <PortalHeader />
      <main className={styles.main}>{children}</main>
    </div>
  );
}
