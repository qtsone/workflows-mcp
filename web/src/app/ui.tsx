import type { ButtonHTMLAttributes, ReactNode } from "react";
import { useId } from "react";

type Tone = "neutral" | "info" | "success" | "warning" | "danger";
type ButtonVariant = "primary" | "secondary" | "danger" | "ghost";

const cx = (...classes: Array<string | false | null | undefined>): string =>
  classes.filter(Boolean).join(" ");

type PageHeaderProps = {
  eyebrow?: string;
  title: string;
  description: string;
  actions?: ReactNode;
};

export function PageHeader({ eyebrow, title, description, actions }: PageHeaderProps): JSX.Element {
  return (
    <header className="page-header">
      <div>
        {eyebrow ? <p className="page-header__eyebrow">{eyebrow}</p> : null}
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {actions ? <div className="page-header__actions">{actions}</div> : null}
    </header>
  );
}

type PanelProps = {
  title: string;
  description?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
};

export function Panel({ title, description, actions, children, className }: PanelProps): JSX.Element {
  const titleId = useId();

  return (
    <section className={cx("ui-panel", className)} role="region" aria-labelledby={titleId}>
      <div className="ui-panel__header">
        <div>
          <h2 id={titleId}>{title}</h2>
          {description ? <p>{description}</p> : null}
        </div>
        {actions ? <div className="ui-panel__actions">{actions}</div> : null}
      </div>
      <div className="ui-panel__body">{children}</div>
    </section>
  );
}

type StatusBadgeProps = {
  tone?: Tone;
  children: ReactNode;
  className?: string;
};

export function StatusBadge({ tone = "neutral", children, className }: StatusBadgeProps): JSX.Element {
  return (
    <span className={cx("status-badge", className)} data-tone={tone}>
      {children}
    </span>
  );
}

type ActionButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
};

export function ActionButton({
  variant = "secondary",
  className,
  type = "button",
  ...props
}: ActionButtonProps): JSX.Element {
  return (
    <button
      {...props}
      type={type}
      className={cx("action-button", `action-button--${variant}`, className)}
    />
  );
}

export function FolderIcon(): JSX.Element {
  return (
    <svg className="ui-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path
        d="M3.75 6.75A2.25 2.25 0 0 1 6 4.5h4.1c.53 0 1.04.21 1.42.59l1.16 1.16H18A2.25 2.25 0 0 1 20.25 8.5v8.75A2.25 2.25 0 0 1 18 19.5H6a2.25 2.25 0 0 1-2.25-2.25V6.75Z"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
      <path
        d="M4.25 9.5h15.5"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}
