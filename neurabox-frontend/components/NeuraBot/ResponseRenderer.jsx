import styles from "./ChatMessage.module.css";

function EventCards({ events }) {
  return (
    <div className={styles.eventGrid}>
      {events.map((event, idx) => (
        <div key={idx} className={styles.eventCard}>
          <h4>{event.title}</h4>
          {event.date && <p className={styles.meta}>📅 {event.date}</p>}
          {event.time && <p className={styles.meta}>🕐 {event.time}</p>}
          {event.raw_text && (
            <p className={styles.meta}>{event.raw_text}</p>
          )}
        </div>
      ))}
    </div>
  );
}

function RichMessage({ blocks }) {
  return (
    <div className={styles.richMessage}>
      {blocks.map((block, idx) => {
        if (block.type === "paragraph") {
          return <p key={idx}>{block.text}</p>;
        }

        if (block.type === "list") {
          return (
            <ul key={idx}>
              {block.items.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          );
        }

        return null;
      })}
    </div>
  );
}

export default function ResponseRenderer({ ui }) {
  if (!ui) return null;

  switch (ui.response_type) {
    case "event_cards":
      return <EventCards events={ui.data.events} />;

    case "rich_message":
      return <RichMessage blocks={ui.data.blocks} />;

    default:
      return <p>Unsupported response</p>;
  }
}
