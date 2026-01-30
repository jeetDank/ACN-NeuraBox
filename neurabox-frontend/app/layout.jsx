import './globals.css';
import ClientLayout from './ClientLayout';

import { Poppins } from 'next/font/google';
const poppins = Poppins({
  subsets: ['latin'],
    weight: ['400', '700'],        // Only load regular and bold
  style: ['normal', 'italic'],   // Load normal and italic styles
  display: 'swap',               // Font display strategy
  variable: '--font-poppins',
})

export const metadata = {
    title: 'ACN Link',
    description: 'One smart prompt connects you to ACN',
};

export default function RootLayout({ children }) {
    return (
        <html lang="en" className={poppins.variable}>
            <body style={{
                margin: 0,
                padding: 0,
                backgroundImage: 'url("/images/acn_landing_page.png")',
                backgroundSize: 'cover',
                backgroundPosition: 'center',
                backgroundRepeat: 'no-repeat',
                overflow: 'hidden'
            }}>
                <ClientLayout>
                    {children}
                </ClientLayout>
            </body>
        </html>
    );
}
