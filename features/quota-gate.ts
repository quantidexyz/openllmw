import type {
  TProviderUsageSnapshot,
  TSubscriptionMeterMatch,
} from "@openllmsh/protocol";
import { matchesSubscriptionMeter } from "@openllmsh/protocol";

/** Maximum age for a stale quota snapshot to remain eligible for routing. */
export const GATE_STALE_CAP_MS = 10 * 60_000;

export type TQuotaGateDecision =
  | { readonly kind: "allow" }
  | { readonly kind: "skip"; readonly reason: string };

type TQuotaPool = Extract<
  TProviderUsageSnapshot,
  { readonly kind: "quota" }
>["windows"][number];

const stalePoolIsGateable = (
  snapshot: Extract<TProviderUsageSnapshot, { readonly kind: "quota" }>,
  pool: TQuotaPool | undefined,
  staleCapMs: number,
  now: number,
): boolean => {
  if (!snapshot.stale) return true;
  if (snapshot.as_of_ms === undefined || now - snapshot.as_of_ms > staleCapMs) {
    return false;
  }
  return (
    pool?.reset_at_ms === null ||
    pool?.reset_at_ms === undefined ||
    now < pool.reset_at_ms
  );
};

export const quotaGateDecision = (params: {
  readonly snapshot: TProviderUsageSnapshot | null;
  readonly meter: TSubscriptionMeterMatch | undefined;
  readonly finalHop: boolean;
  readonly staleCapMs: number;
  readonly now: number;
}): TQuotaGateDecision => {
  if (params.finalHop || params.snapshot?.kind !== "quota") {
    return { kind: "allow" };
  }

  const { snapshot } = params;
  const staleProviderRejectedPastReset =
    snapshot.stale &&
    snapshot.windows.some(
      (window) =>
        window.reset_at_ms !== null && params.now >= window.reset_at_ms,
    );
  if (
    snapshot.status === "rejected" &&
    !staleProviderRejectedPastReset &&
    stalePoolIsGateable(
      snapshot,
      snapshot.windows.find((window) => window.percent_used >= 100),
      params.staleCapMs,
      params.now,
    )
  ) {
    return { kind: "skip", reason: "quota_skip: provider window exhausted" };
  }

  const exhaustedPool = snapshot.extra_pools?.find(
    (pool) =>
      matchesSubscriptionMeter(params.meter, pool.meter_id) &&
      pool.percent_used >= 100 &&
      stalePoolIsGateable(snapshot, pool, params.staleCapMs, params.now),
  );
  if (exhaustedPool !== undefined) {
    return { kind: "skip", reason: "quota_skip: model meter exhausted" };
  }

  return { kind: "allow" };
};
