import type { Metadata } from "next";
import { StudyDashboard } from "@/features/portal/components/StudyDashboard";

export const metadata: Metadata = { title: "Studies · Ether" };

export default function StudiesPage() {
  return <StudyDashboard />;
}
