export const JAPAN_TIME_ZONE = "Asia/Tokyo";

export function formatDateInJapan(
	date: Date,
	locale: string,
	options: Intl.DateTimeFormatOptions,
): string {
	return new Intl.DateTimeFormat(locale, {
		...options,
		timeZone: JAPAN_TIME_ZONE,
	}).format(date);
}

function getJapanDatePart(
	date: Date,
	partType: "day" | "month" | "year",
): string {
	const formatter = new Intl.DateTimeFormat("en-US", {
		day: "2-digit",
		month: "2-digit",
		timeZone: JAPAN_TIME_ZONE,
		year: "numeric",
	});
	const part = formatter
		.formatToParts(date)
		.find(({ type }) => type === partType);
	return part?.value ?? "";
}

export function getJapanYear(date: Date): number {
	return Number(getJapanDatePart(date, "year"));
}

export function getJapanMonthIndex(date: Date): number {
	return Number(getJapanDatePart(date, "month")) - 1;
}

export function formatVersionDate(date: Date): string {
	const year = getJapanDatePart(date, "year");
	const month = getJapanDatePart(date, "month");
	const day = getJapanDatePart(date, "day");
	return `${year}.${month}.${day}`;
}
