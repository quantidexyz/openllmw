/**
 * The canonical content parts the inbound adapters PRODUCE — the
 * OpenAI-native subset of `ContentPart` in `packages/protocol/chat.ts`
 * (no `input_audio`: nothing inbound maps to it). Shared by the
 * `adapters/messages` and `adapters/responses` request adapters so the
 * two can't drift apart.
 */
export type TCanonicalContentPart =
  | { readonly type: "text"; readonly text: string }
  | {
      readonly type: "image_url";
      readonly image_url: {
        readonly url: string;
        readonly detail?: "auto" | "low" | "high";
      };
    }
  | {
      readonly type: "file";
      readonly file: {
        readonly file_data?: string;
        readonly file_id?: string;
        readonly filename?: string;
      };
    };
