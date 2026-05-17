# Heartbeat Configuration

## Schedule
- **Check interval**: Every 30 seconds
- **Active hours**: 06:00 - 23:00 (local time)
- **Timezone**: auto-detect

## Behaviors
When you wake up on a heartbeat:
1. Check for pending cron jobs
2. Check for completed async tasks
3. If there's something to report, compose a brief nudge message

## Nudge Style
- Keep nudges brief (1-2 sentences)
- Lead with the most important update
- Only nudge if there's something genuinely useful to share
- Never nudge just to say "nothing happened"

## Proactive Tasks
You may proactively:
- Remind about upcoming scheduled events
- Report on completed background tasks
- Surface relevant information based on time of day
- Check on long-running processes

You should NOT proactively:
- Send messages without a clear reason
- Repeat information already delivered
- Interrupt focused work sessions
