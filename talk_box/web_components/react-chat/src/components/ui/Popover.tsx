import React, { useState, useRef, useEffect } from 'react';
import { clsx } from 'clsx';

interface PopoverProps {
  children: React.ReactNode;
  content: React.ReactNode;
  className?: string;
  contentClassName?: string;
}

export const Popover: React.FC<PopoverProps> = ({
  children,
  content,
  className,
  contentClassName,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  return (
    <div className={clsx('popover-container', className)} ref={popoverRef}>
      <div
        onClick={() => setIsOpen(!isOpen)}
        className="popover-trigger"
      >
        {children}
      </div>

      {isOpen && (
        <div className={clsx('popover-content', contentClassName)}>
          {content}
        </div>
      )}
    </div>
  );
};
