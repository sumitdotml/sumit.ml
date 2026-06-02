import { formatDateInJapan, getJapanYear } from "./dates";

export interface WeeklyRouteParts {
	week: number;
	date: Date;
}

export function formatWeeklyNumber(week: number): string {
	return String(week).padStart(2, "0");
}

export function getWeeklySlug({ date, week }: WeeklyRouteParts): string {
	return `${getJapanYear(date)}-w${formatWeeklyNumber(week)}`;
}

export function getWeeklyHref(parts: WeeklyRouteParts): string {
	return `/weekly/${getWeeklySlug(parts)}/`;
}

export function formatWeeklyTitle({ date, week }: WeeklyRouteParts): string {
	return `Week ${week}, ${getJapanYear(date)}`;
}

export function formatWeeklyMonth(date: Date): string {
	return formatDateInJapan(date, "en-US", { month: "long" });
}
