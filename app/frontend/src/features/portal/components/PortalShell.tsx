import Link from "next/link";
import type { ReactNode } from "react";
import styles from "./PortalShell.module.css";

export const DISCLAIMER =
  "Decision support only. This is not a diagnosis; a qualified radiologist must review every result.";

/** Frame of every portal page: brand header and the always-visible disclaimer. */
export function PortalShell({ children }: { children: ReactNode }) {
  return (
    <div className="portal">
      <header className={styles.header}>
        <Link href="/" className={styles.brand}>
          Ether
        </Link>
        <Link href="/studies" className={styles.nav}>
          Studies
        </Link>
      </header>
      <p className={styles.disclaimer} role="note">
        {DISCLAIMER}
      </p>
      <main className={styles.main}>{children}</main>
    </div>
  );
}
