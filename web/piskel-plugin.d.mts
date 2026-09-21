import type { Plugin } from "vite";

export const VENDOR: string;
export const OUT: string;
export const PLACEHOLDER: string;
export function piskelBody(): string;
export function generate(): string[];
export default function piskel(): Plugin;
