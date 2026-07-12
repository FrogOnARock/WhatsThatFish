/* Small confirm dialog for destructive, irreversible actions (deletes). Matches
   the app's modal styling; the confirm button carries the coral (danger) accent.
   Shows a busy state and surfaces an error inline rather than closing on failure. */
import { useState } from "react";

interface Props {
  title: string;
  body: string;
  confirmLabel?: string;
  onConfirm: () => Promise<void>;
  onClose: () => void;
}

export default function ConfirmModal({
  title,
  body,
  confirmLabel = "Delete",
  onConfirm,
  onClose,
}: Props) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleConfirm() {
    setBusy(true);
    setError(null);
    try {
      await onConfirm();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="modal__backdrop" onClick={busy ? undefined : onClose}>
      <div className="modal modal--sm" onClick={(e) => e.stopPropagation()}>
        <h3 className="modal__title">{title}</h3>
        <p className="modal__body">{body}</p>
        {error && <div className="modal__error">{error}</div>}
        <div className="modal__actions">
          <button className="btn btn--ghost btn--sm" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button
            className="btn btn--coral btn--sm"
            onClick={handleConfirm}
            disabled={busy}
          >
            {busy ? "Deleting…" : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
