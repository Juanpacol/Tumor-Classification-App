"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import styles from "./PortalHeader.module.css";

export const DISCLAIMER =
  "Decision support only. This is not a diagnosis; a qualified radiologist must review every result.";

/** Brand, navigation and the always-visible disclaimer. */
export function PortalHeader() {
  const pathname = usePathname();
  const inStudies = pathname?.startsWith("/studies") ?? false;

  return (
    <header className={styles.bar}>
      <div className={styles.row}>
        <Link href="/" className={styles.brand}>
          Ether
        </Link>
        <nav aria-label="Portal">
          <Link
            href="/studies"
            className={styles.link}
            aria-current={inStudies ? "page" : undefined}
          >
            Studies
          </Link>
        </nav>
      </div>
      <p className={styles.disclaimer} role="note">
        {DISCLAIMER}
      </p>
    </header>
  );
}
