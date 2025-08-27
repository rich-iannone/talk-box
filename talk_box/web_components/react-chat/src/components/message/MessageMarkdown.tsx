import React, { Suspense } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import rehypeHighlight from 'rehype-highlight';
import { clsx } from 'clsx';

interface MessageMarkdownProps {
  children: string;
  className?: string;
}

const MessageMarkdown: React.FC<MessageMarkdownProps> = ({ children, className }) => {
  return (
    <div className={clsx('prose prose-sm max-w-none', className)}>
      <Suspense fallback={<div>Loading...</div>}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkBreaks]}
          rehypePlugins={[rehypeHighlight]}
          components={{
            // Customize how different elements are rendered
            code: ({ className, children, ...props }) => {
              return (
                <code className={className} {...props}>
                  {children}
                </code>
              );
            },
            // Style tables
            table: ({ children }) => (
              <div className="overflow-x-auto">
                <table className="min-w-full border-collapse border border-gray-300">
                  {children}
                </table>
              </div>
            ),
            th: ({ children }) => (
              <th className="border border-gray-300 px-4 py-2 bg-gray-50 font-semibold text-left">
                {children}
              </th>
            ),
            td: ({ children }) => (
              <td className="border border-gray-300 px-4 py-2">
                {children}
              </td>
            ),
            // Style links
            a: ({ href, children }) => (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:text-blue-800 underline"
              >
                {children}
              </a>
            ),
            // Style blockquotes
            blockquote: ({ children }) => (
              <blockquote className="border-l-4 border-gray-300 pl-4 italic text-gray-700">
                {children}
              </blockquote>
            ),
          }}
        >
          {children}
        </ReactMarkdown>
      </Suspense>
    </div>
  );
};

export default MessageMarkdown;
