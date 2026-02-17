import type { SimulationState, WorldInteractionResponse } from "../../types";

interface Props {
  simulation: SimulationState | null;
  interaction: WorldInteractionResponse | null;
  interactingPersonId: string | null;
  onInteract: (personId: string, action: "speak" | "pray" | "sing") => void;
}

export function MythWorldPanel({
  simulation,
  interaction,
  interactingPersonId,
  onInteract,
}: Props) {
  const world = simulation?.world;

  if (!world) {
    return (
      <div className="text-muted text-sm p-4">
        World pulse is gathering / 世界の脈動を収集中
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-3">
        {world.people.map((person) => (
          <article
            key={person.id}
            className="border border-line rounded-xl bg-white/70 p-3"
          >
            <div className="flex items-baseline justify-between">
              <strong className="text-sm">{person.name.en}</strong>
              <span className="text-xs text-muted">{person.name.ja}</span>
            </div>
            <p className="text-xs text-muted mt-1">
              {person.role.en} / {person.role.ja}
            </p>
            <p className="text-xs mt-2">
              {person.instrument} · {person.hymn_bpm} BPM
            </p>
            <p className="text-xs text-muted mt-1">prays to {person.prays_to}</p>
            <div className="mt-2 h-2 bg-bg-1 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500"
                style={{ width: `${Math.max(4, Math.round(person.prayer_intensity * 100))}%` }}
              />
            </div>
            <div className="mt-3 flex gap-2">
              <button
                type="button"
                onClick={() => onInteract(person.id, "speak")}
                disabled={interactingPersonId === person.id}
                className="text-[10px] px-2 py-1 rounded border border-line bg-white/70 hover:bg-white disabled:opacity-60"
              >
                Speak / 話す
              </button>
              <button
                type="button"
                onClick={() => onInteract(person.id, "pray")}
                disabled={interactingPersonId === person.id}
                className="text-[10px] px-2 py-1 rounded border border-line bg-white/70 hover:bg-white disabled:opacity-60"
              >
                Pray / 祈る
              </button>
              <button
                type="button"
                onClick={() => onInteract(person.id, "sing")}
                disabled={interactingPersonId === person.id}
                className="text-[10px] px-2 py-1 rounded border border-line bg-white/70 hover:bg-white disabled:opacity-60"
              >
                Sing / 歌う
              </button>
            </div>
          </article>
        ))}
      </div>

      {interaction?.ok && (
        <section className="border border-line rounded-xl bg-sky-50/70 p-3">
          <h4 className="font-bold text-sm mb-2">Presence Dialogue / プレゼンス対話</h4>
          <p className="text-xs font-semibold">
            {interaction.speaker?.en} / {interaction.speaker?.ja}
          </p>
          <p className="text-xs text-muted mt-1">
            {interaction.presence?.name.en} / {interaction.presence?.name.ja}
          </p>
          <p className="text-xs mt-2">{interaction.line_en}</p>
          <p className="text-xs text-muted mt-1">{interaction.line_ja}</p>
        </section>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <section className="border border-line rounded-xl bg-white/55 p-3">
          <h4 className="font-bold text-sm mb-2">Songbook / 聖歌帳</h4>
          <div className="space-y-2">
            {world.songs.slice(0, 4).map((song) => (
              <div key={song.id} className="text-xs border-l-2 border-blue-200 pl-2">
                <div className="font-semibold">{song.title.en}</div>
                <div className="text-muted">{song.title.ja}</div>
                <div>{song.bpm} BPM · energy {song.energy.toFixed(2)}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="border border-line rounded-xl bg-white/55 p-3">
          <h4 className="font-bold text-sm mb-2">Library / 文庫</h4>
          <div className="space-y-2">
            {world.books.length === 0 && (
              <p className="text-xs text-muted">Scribes are drafting the first myth.</p>
            )}
            {world.books.slice(-4).reverse().map((book) => (
              <div key={book.id} className="text-xs border-l-2 border-amber-300 pl-2">
                <div className="font-semibold">{book.title.en}</div>
                <div className="text-muted">{book.title.ja}</div>
                <div className="text-muted">by {book.author.en}</div>
                <div>{book.excerpt.en}</div>
              </div>
            ))}
          </div>
        </section>
      </div>

      <p className="text-xs text-muted">
        prayer intensity {world.prayer_intensity.toFixed(2)} · tick {world.tick}
      </p>
    </div>
  );
}
