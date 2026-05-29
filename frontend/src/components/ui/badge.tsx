import { clsx } from "clsx";
import type { HTMLAttributes } from "react";

export function Badge({ className, ...props }: HTMLAttributes<HTMLSpanElement>) {
  return <span className={clsx("ui-badge", className)} {...props} />;
}
