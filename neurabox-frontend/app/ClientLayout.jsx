'use client';

import { ThemeProvider } from '@/hooks/useTheme';

export default function ClientLayout({ children }) {
    return <ThemeProvider>{children}</ThemeProvider>;
}
