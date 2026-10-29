import type { Metadata } from "next";
import { StudyWorkspace } from "@/features/portal/components/StudyWorkspace";

export const metadata: Metadata = { title: "Study · Ether" };

export default function StudyPage({ params }: { params: { id: string } }) {
  return <StudyWorkspace id={params.id} />;
}
