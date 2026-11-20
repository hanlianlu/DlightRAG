// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The Web command catalog: one descriptor per slash command, and the words a kept command's result
 *  shows.
 *
 * A command owns its name, the words that describe it, and how its structured result is rendered.
 * The composer's slash menu reads this catalog; the server executes the command and keeps the
 * command event, so no browser code runs a command handler. The server keeps a structured
 * `result_data` beside the canonical English `result`, and this module is the one place that renders
 * that structure through lit-localize. */

import {msg, str} from '@lit/localize';
import type {ConversationCommand} from '../api/conversations.ts';
import {goalPhaseLabel} from './goal-phase.ts';
import {catalogText, localizedText, serverText, type LocalizedText} from './localized-text.ts';

/** What a command's result node draws, resolved each time it draws. One result can be several
 *  localized lines (a goal state is: its phase, objective, round, and so on), so a view holds this
 *  thunk and the component resolves it while drawing: a language change then redraws every line,
 *  and nothing under lib/ settles words a later switch would leave behind. */
export type CommandResultText = () => string;

export interface WebCommand {
  readonly name: string;
  readonly description: LocalizedText;
  readonly inputHint: LocalizedText;
  /** The words a kept command's structured result shows, when the command keeps one. Null means this
   *  client cannot read the payload, and the server's canonical English `result` stands in. */
  resultText?(resultData: Record<string, unknown> | null): CommandResultText | null;
}

export interface CommandInvocation {
  readonly command: WebCommand;
  readonly args: string;
}

const goalCommand: WebCommand = {
  name: 'goal',
  description: catalogText(
    'Set or view the goal for a long-running task',
    'goalCommand.description',
  ),
  inputHint: catalogText(
    '[<objective>|cap N|edit <objective>|pause|resume|clear]',
    'goalCommand.inputHint',
  ),
  resultText: goalResultText,
};

const WEB_COMMANDS: readonly WebCommand[] = [goalCommand];

/** The command one complete slash line names, with the words after its name. */
export function findWebCommand(text: string): CommandInvocation | null {
  const match = /^\/([a-z0-9-]+)(?:\s+([\s\S]*))?$/.exec(text.trim());
  if (match === null) return null;
  const command = WEB_COMMANDS.find((candidate) => candidate.name === match[1]);
  return command === undefined ? null : {command, args: (match[2] ?? '').trim()};
}

/** The commands whose name still prefixes the reader's bare `/word`. */
export function commandSuggestions(text: string): readonly WebCommand[] {
  const raw = text.trimStart();
  if (/\s/.test(raw)) return [];
  const match = /^\/([a-z0-9-]*)$/.exec(raw);
  if (match === null) return [];
  const prefix = match[1] ?? '';
  return WEB_COMMANDS.filter((command) => command.name.startsWith(prefix));
}

/** The words one persisted command result shows. A command whose catalog entry can read the
 *  structured `result_data` renders it through lit-localize in the language showing now; the
 *  server's canonical English `result` is the fallback only when that payload is absent or a shape
 *  this client cannot read. */
export function commandResultText(command: ConversationCommand): CommandResultText {
  const catalogued = WEB_COMMANDS.find((candidate) => candidate.name === command.name);
  const clientText = catalogued?.resultText?.(command.resultData) ?? null;
  return clientText ?? (() => localizedText(serverText(command.result)));
}

const GOAL_PHASES = ['active', 'paused', 'blocked', 'complete'] as const;

/** The state a `goal_status` result describes, as the client reads it. */
interface GoalResultState {
  readonly objective: string;
  readonly phase: (typeof GOAL_PHASES)[number];
  readonly maxRounds: number;
  readonly roundsStarted: number;
  readonly blockedReason: string | null;
}

/** Read the goal state a structured result names, or null for a payload this client does not fully
 *  know: a missing or mistyped fact means the canonical English result is the honest one to show. */
function readGoalResultState(value: unknown): GoalResultState | null {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const {objective, phase, max_rounds, rounds_started, blocked_reason} = record;
  if (typeof objective !== 'string') return null;
  if (typeof max_rounds !== 'number' || !Number.isSafeInteger(max_rounds)) return null;
  if (typeof rounds_started !== 'number' || !Number.isSafeInteger(rounds_started)) return null;
  if (!GOAL_PHASES.includes(phase as GoalResultState['phase'])) return null;
  if (blocked_reason !== null && typeof blocked_reason !== 'string') return null;
  return {
    objective,
    phase: phase as GoalResultState['phase'],
    maxRounds: max_rounds,
    roundsStarted: rounds_started,
    blockedReason: blocked_reason,
  };
}

/** The words one goal command's structured result shows, or null when the payload states a shape
 *  this client cannot render (an older server, a goal fact added since this client was built). */
function goalResultText(resultData: Record<string, unknown> | null): CommandResultText | null {
  if (resultData === null) return null;
  switch (resultData.kind) {
    case 'goal_cleared':
      return () => msg('Goal cleared.', {id: 'goalCommand.cleared'});
    case 'goal_invalid':
      if (resultData.reason === 'cap') {
        return () => msg('Add a positive number: /goal cap 5', {id: 'goalCommand.invalidCap'});
      }
      if (resultData.reason === 'objective') {
        return () => msg('Add an objective: /goal <objective>', {id: 'goalCommand.invalidEdit'});
      }
      return null;
    case 'goal_error': {
      // The stable code gives the reader a localized line; the refusal's own sentence states the
      // actual condition ("pause needs an active goal; this one is complete"), so it is appended
      // rather than hidden. A payload without either falls through to the server's canonical
      // result, which is the same sentence.
      const message = typeof resultData.message === 'string' && resultData.message.trim() !== ''
        ? resultData.message
        : null;
      const localized = goalCommandErrorText(
        typeof resultData.code === 'string' ? resultData.code : null,
      );
      if (localized !== null) {
        return () => {
          const head = localizedText(localized);
          return message === null || message === localized.source ? head : `${head} ${message}`;
        };
      }
      if (message === null) return null;
      return () => localizedText(serverText(message));
    }
    case 'goal_status':
      return goalStatusText(resultData.goal);
    default:
      return null;
  }
}

/** The localized words one goal refusal's stable code names, or null when this client does not
 *  know the code: the caller then shows the server's own sentence, so a later domain reason is
 *  still readable rather than replaced by a generic line. */
export function goalCommandErrorText(code: string | null): LocalizedText | null {
  switch (code) {
    case 'goal_not_found':
      return catalogText('This conversation carries no goal.', 'goalCommand.error.goalNotFound');
    case 'goal_invalid_transition':
      return catalogText(
        'The goal cannot make that change now.',
        'goalCommand.error.goalInvalidTransition',
      );
    case 'goal_rounds_exhausted':
      return catalogText(
        'All rounds have started. Raise the cap before resuming.',
        'goalCommand.error.goalRoundsExhausted',
      );
    case 'goal_invalid_objective':
      return catalogText('The goal needs a valid objective.', 'goalCommand.error.goalInvalidObjective');
    case 'goal_invalid_rounds':
      return catalogText(
        'The round cap must be a positive whole number.',
        'goalCommand.error.goalInvalidRounds',
      );
    case 'goal_stale_revision':
      return catalogText('The goal changed while it was read. Try again.', 'goalCommand.error.goalStale');
    case 'goal_invalid_block_reason':
      return catalogText(
        'A blocked goal needs the concrete condition.',
        'goalCommand.error.goalInvalidBlockReason',
      );
    default:
      return null;
  }
}

/** The line a goal state shows: its phase and round. The objective, the blocked condition and the
 *  controls live in the goal bar, which is the state's home; a transcript line repeating them would
 *  be a card the reader has to scroll past. */
function goalStatusText(goal: unknown): CommandResultText | null {
  if (goal === null) {
    return () => msg('No goal in this conversation. /goal <objective> creates one.', {
      id: 'goalCommand.none',
    });
  }
  const state = readGoalResultState(goal);
  if (state === null) return null;
  return () => {
    const head = `${goalPhaseLabel(state)} · ${msg(
      str`Round: ${state.roundsStarted}/${state.maxRounds}`,
      {id: 'goalCommand.round'},
    )}`;
    // A paused goal ends the round already running and starts no next one; the receipt says so.
    return state.phase === 'paused'
      ? `${head} · ${msg(
        'Paused. The goal round in progress was cancelled; no next round will start. Your own running turn is not stopped.',
        {id: 'goalCommand.pauseNote'},
      )}`
      : head;
  };
}
