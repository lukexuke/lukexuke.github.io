(() => {
  // Theme, navigation, and publication layout remain independent.
  const themeButtons = [...document.querySelectorAll('.theme-toggle')];
  const applyTheme = (theme, save = false) => {
    const dark = theme === 'dark';
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
    themeButtons.forEach(button => {
      button.setAttribute('aria-pressed', String(dark));
      button.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
    });
    if (save) {
      try { localStorage.setItem('kexu-theme', dark ? 'dark' : 'light'); }
      catch (_) { /* The toggle still works when storage is unavailable. */ }
    }
  };
  applyTheme(document.documentElement.getAttribute('data-theme'));
  themeButtons.forEach(button => button.addEventListener('click', () => {
    applyTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark', true);
  }));
  const navigation = document.querySelector('.primary-nav');
  const toggle = document.querySelector('.compact-menu');
  const navLinks = [...navigation.querySelectorAll('a[href^="#"]')];
  const mobile = window.matchMedia('(max-width: 1080px)');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const closeMenu = () => {
    navigation.classList.remove('is-open');
    toggle.setAttribute('aria-expanded', 'false');
    toggle.setAttribute('aria-label', 'Open navigation');
  };
  toggle.addEventListener('click', () => {
    const open = toggle.getAttribute('aria-expanded') !== 'true';
    navigation.classList.toggle('is-open', open);
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
  });
  navLinks.forEach(link => link.addEventListener('click', closeMenu));
  mobile.addEventListener('change', closeMenu);
  document.addEventListener('click', event => { if (!event.target.closest('.masthead')) closeMenu(); });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && toggle.getAttribute('aria-expanded') === 'true') {
      closeMenu(); toggle.focus();
    }
  });
  const positionTeaching = () => {
    const section = document.getElementById('teach');
    section.style.removeProperty('margin-bottom');
    const top = section.getBoundingClientRect().top + window.scrollY;
    const headerHeight = document.querySelector('.masthead').offsetHeight;
    const visibleHeight = window.innerHeight - headerHeight;
    const desiredTop = Math.max(headerHeight + 16, headerHeight + (visibleHeight - section.offsetHeight) / 2);
    window.scrollTo({top: Math.max(0, top - desiredTop), behavior: reducedMotion.matches ? 'instant' : 'smooth'});
  };
  document.querySelectorAll('a[href="#teach"]').forEach(link => link.addEventListener('click', event => {
    event.preventDefault(); closeMenu();
    if (location.hash !== '#teach') history.pushState(null, '', '#teach');
    positionTeaching();
  }));
  window.addEventListener('hashchange', () => {
    if (location.hash === '#teach') positionTeaching();
  });
  window.addEventListener('resize', () => { if (location.hash === '#teach') positionTeaching(); }, {passive:true});
  window.addEventListener('load', () => { if (location.hash === '#teach') positionTeaching(); }, {once:true});
  const grid = document.querySelector('.papers-grid');
  const viewButtons = [...document.querySelectorAll('[data-view]')];
  const setView = (view, save = false) => {
    grid.classList.toggle('is-list', view === 'list');
    viewButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.view === view)));
    if (save) { try { localStorage.setItem('kexu-publication-view', view); } catch (_) {} }
  };
  try {
    const savedView = localStorage.getItem('kexu-publication-view');
    if (savedView === 'grid' || savedView === 'list') setView(savedView);
  } catch (_) {}
  viewButtons.forEach(button => button.addEventListener('click', () => setView(button.dataset.view, true)));
  const sections = navLinks.map(link => document.querySelector(link.hash)).filter(Boolean);
  let scheduled = false;
  const updateActive = () => {
    scheduled = false;
    const boundary = window.innerHeight * .35;
    let current = sections[0], nearestTop = -Infinity;
    for (const section of sections) {
      const top = section.getBoundingClientRect().top;
      if (top > boundary) continue;
      if (top > nearestTop + 2 || (Math.abs(top - nearestTop) <= 2 && location.hash === `#${section.id}`)) {
        current = section; nearestTop = top;
      }
    }
    navLinks.forEach(link => {
      const active = link.hash === `#${current.id}`;
      link.classList.toggle('is-active', active);
      if (active) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  };
  window.addEventListener('scroll', () => {
    if (!scheduled) { scheduled = true; requestAnimationFrame(updateActive); }
  }, {passive:true});
  window.addEventListener('resize', updateActive, {passive:true});
  updateActive();
})();
