import type { ReactNode } from "react";
import { PortalShell } from "@/features/portal/components/PortalShell";

export default function StudiesLayout({ children }: { children: ReactNode }) {
  return <PortalShell>{children}</PortalShell>;
}
