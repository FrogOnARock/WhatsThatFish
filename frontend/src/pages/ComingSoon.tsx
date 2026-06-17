interface ComingSoonProps {
  title: string;
  hint: string;
}

export default function ComingSoon({ title, hint }: ComingSoonProps) {
  return (
    <main className="main">
      <div className="main__inner">
        <header className="page-header">
          <div>
            <div className="page-header__crumb">Workspace</div>
            <h1 className="page-header__title">{title}</h1>
            <p className="page-header__subtitle">{hint}</p>
          </div>
          <div className="page-header__model">
            <span className="page-header__model-pill">not yet built</span>
          </div>
        </header>
        <div className="coming-soon">
          <div className="coming-soon__inner">
            <div className="coming-soon__mark" />
            <p>This page is on the roadmap.</p>
          </div>
        </div>
      </div>
    </main>
  );
}
