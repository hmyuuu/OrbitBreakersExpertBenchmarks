(() => {
  const root = document.documentElement;

  const setTheme = (theme) => {
    root.dataset.theme = theme;
    localStorage.setItem("ob-theme", theme);
  };

  document.querySelector(".theme-toggle")?.addEventListener("click", () => {
    setTheme(root.dataset.theme === "dark" ? "light" : "dark");
  });

  const sidebarToggle = document.querySelector(".sidebar-toggle");
  const sidebar = document.querySelector("#docs-sidebar");
  const sidebarMedia = window.matchMedia("(max-width: 980px)");
  const syncSidebarState = () => {
    if (!sidebar || !sidebarToggle) return;
    if (!sidebarMedia.matches) {
      document.body.classList.remove("sidebar-open");
      sidebar.inert = false;
      sidebar.removeAttribute("aria-hidden");
      sidebarToggle.setAttribute("aria-expanded", "false");
      sidebarToggle.setAttribute("aria-label", "Open task navigation");
      return;
    }
    const open = document.body.classList.contains("sidebar-open");
    sidebar.inert = !open;
    sidebar.setAttribute("aria-hidden", String(!open));
    sidebarToggle.setAttribute("aria-expanded", String(open));
    sidebarToggle.setAttribute(
      "aria-label",
      open ? "Close task navigation" : "Open task navigation",
    );
  };
  const closeSidebar = (restoreFocus = true) => {
    document.body.classList.remove("sidebar-open");
    if (restoreFocus) sidebarToggle?.focus();
    syncSidebarState();
  };

  sidebarToggle?.addEventListener("click", () => {
    const open = document.body.classList.toggle("sidebar-open");
    syncSidebarState();
    if (open) {
      window.setTimeout(() => sidebar?.querySelector("a")?.focus(), 0);
    }
  });
  document.querySelector("[data-close-sidebar]")?.addEventListener("click", closeSidebar);
  document.querySelectorAll(".docs-sidebar a").forEach((link) => {
    link.addEventListener("click", () => closeSidebar(false));
  });
  sidebarMedia.addEventListener("change", syncSidebarState);
  syncSidebarState();

  const filterButtons = [...document.querySelectorAll("[data-filter]")];
  const filterable = [
    ...document.querySelectorAll(".task-table tbody tr[data-task-kind], .task-card-mobile[data-task-kind]"),
  ];
  const filterRows = [...document.querySelectorAll(".task-table tbody tr[data-task-kind]")];
  const filterCount = document.querySelector("[data-filter-count]");
  const matchesFilter = (kind, filter) => {
    if (filter === "all") return true;
    if (filter === "docker") return kind === "validated" || kind === "caveat";
    if (filter === "qualified") return kind === "qualified";
    if (filter === "local") return kind === "provisional";
    if (filter === "special") return kind === "caveat" || kind === "feasibility";
    return true;
  };

  filterButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const filter = button.dataset.filter;
      filterButtons.forEach((item) => {
        const active = item === button;
        item.classList.toggle("is-active", active);
        item.setAttribute("aria-pressed", String(active));
      });
      filterable.forEach((item) => {
        item.hidden = !matchesFilter(item.dataset.taskKind, filter);
      });
      if (filterCount) {
        filterCount.textContent = String(
          filterRows.filter((item) => !item.hidden).length,
        );
      }
    });
  });

  const graphTabs = [...document.querySelectorAll("[data-graph-tab]")];
  const activateGraphTab = (tab, focus = false) => {
    graphTabs.forEach((item) => {
      const active = item === tab;
      item.setAttribute("aria-selected", String(active));
      item.tabIndex = active ? 0 : -1;
      const panel = document.getElementById(item.getAttribute("aria-controls"));
      if (panel) panel.hidden = !active;
    });
    if (focus) tab.focus();
  };

  graphTabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activateGraphTab(tab));
    tab.addEventListener("keydown", (event) => {
      let nextIndex = null;
      if (event.key === "ArrowRight") nextIndex = (index + 1) % graphTabs.length;
      if (event.key === "ArrowLeft") nextIndex = (index - 1 + graphTabs.length) % graphTabs.length;
      if (event.key === "Home") nextIndex = 0;
      if (event.key === "End") nextIndex = graphTabs.length - 1;
      if (nextIndex === null) return;
      event.preventDefault();
      activateGraphTab(graphTabs[nextIndex], true);
    });
  });

  if (window.location.hash.startsWith("#graph-panel-")) {
    const deepLinkTab = graphTabs.find(
      (tab) => `#${tab.getAttribute("aria-controls")}` === window.location.hash,
    );
    if (deepLinkTab) activateGraphTab(deepLinkTab);
  }

  const dialog = document.querySelector("#search-dialog");
  const input = document.querySelector("#task-search");
  const results = document.querySelector("#search-results");
  const empty = document.querySelector("#search-empty");
  const dataNode = document.querySelector("#task-search-data");
  const tasks = dataNode ? JSON.parse(dataNode.textContent) : [];
  let selectedIndex = 0;
  let visibleTasks = tasks;
  let searchOpener = null;

  const normalize = (value) =>
    value
      .toLocaleLowerCase()
      .normalize("NFKD")
      .replace(/\p{Diacritic}/gu, "");

  const renderSearch = (query = "") => {
    if (!results) return;
    const term = normalize(query.trim());
    visibleTasks = tasks.filter((task) =>
      normalize(`${task.id} ${task.title} ${task.summary} ${task.metric} ${task.status}`).includes(term),
    );
    selectedIndex = Math.min(selectedIndex, Math.max(0, visibleTasks.length - 1));
    results.replaceChildren();

    visibleTasks.forEach((task, index) => {
      const link = document.createElement("a");
      link.className = `search-result${index === selectedIndex ? " is-selected" : ""}`;
      link.id = `search-option-${task.id}`;
      link.href = task.href;
      link.setAttribute("role", "option");
      link.setAttribute("aria-selected", String(index === selectedIndex));

      const id = document.createElement("span");
      id.className = "task-number";
      id.textContent = task.id;

      const copy = document.createElement("span");
      const title = document.createElement("strong");
      title.textContent = task.title;
      const summary = document.createElement("small");
      summary.textContent = `${task.status} · ${task.summary}`;
      copy.append(title, summary);

      const metric = document.createElement("em");
      metric.textContent = task.metric;
      link.append(id, copy, metric);
      results.append(link);
    });

    input?.setAttribute(
      "aria-activedescendant",
      visibleTasks[selectedIndex] ? `search-option-${visibleTasks[selectedIndex].id}` : "",
    );
    if (empty) empty.hidden = visibleTasks.length !== 0;
  };

  const openSearch = (opener = null) => {
    if (!dialog) return;
    searchOpener = opener || document.activeElement;
    renderSearch(input?.value || "");
    if (typeof dialog.showModal === "function") dialog.showModal();
    input?.setAttribute("aria-expanded", "true");
    input?.focus();
  };

  const closeSearch = () => {
    if (dialog?.open) dialog.close();
    input?.setAttribute("aria-expanded", "false");
    if (searchOpener instanceof HTMLElement) searchOpener.focus();
  };

  document.querySelectorAll("[data-open-search]").forEach((button) => {
    button.addEventListener("click", () => openSearch(button));
  });
  document.querySelector("[data-close-search]")?.addEventListener("click", closeSearch);
  input?.addEventListener("input", () => renderSearch(input.value));

  dialog?.addEventListener("click", (event) => {
    if (event.target === dialog) closeSearch();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && document.body.classList.contains("sidebar-open")) {
      event.preventDefault();
      closeSidebar();
      return;
    }

    if ((event.metaKey || event.ctrlKey) && event.key.toLocaleLowerCase() === "k") {
      event.preventDefault();
      dialog?.open ? closeSearch() : openSearch(document.activeElement);
      return;
    }

    if (!dialog?.open) return;
    if (event.key === "Escape") {
      event.preventDefault();
      closeSearch();
    } else if (event.key === "ArrowDown" && visibleTasks.length) {
      event.preventDefault();
      selectedIndex = (selectedIndex + 1) % visibleTasks.length;
      renderSearch(input?.value || "");
      results?.children[selectedIndex]?.scrollIntoView({ block: "nearest" });
    } else if (event.key === "ArrowUp" && visibleTasks.length) {
      event.preventDefault();
      selectedIndex = (selectedIndex - 1 + visibleTasks.length) % visibleTasks.length;
      renderSearch(input?.value || "");
      results?.children[selectedIndex]?.scrollIntoView({ block: "nearest" });
    } else if (event.key === "Enter" && visibleTasks[selectedIndex]) {
      event.preventDefault();
      window.location.href = visibleTasks[selectedIndex].href;
    }
  });

  renderSearch();

  document.querySelectorAll("[data-copy-target]").forEach((button) => {
    button.addEventListener("click", async () => {
      const target = document.getElementById(button.dataset.copyTarget);
      if (!target) return;
      const original = button.textContent;
      try {
        await navigator.clipboard.writeText(target.textContent);
        button.textContent = "Copied";
      } catch {
        button.textContent = "Copy failed";
      }
      window.setTimeout(() => {
        button.textContent = original;
      }, 1500);
    });
  });

  const tocLinks = [...document.querySelectorAll(".docs-toc > a")];
  const observedSections = tocLinks
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  if (tocLinks.length && "IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (!visible.length) return;
        const id = `#${visible[0].target.id}`;
        tocLinks.forEach((link) => link.classList.toggle("is-active", link.getAttribute("href") === id));
      },
      { rootMargin: "-15% 0px -72% 0px", threshold: [0, 1] },
    );
    observedSections.forEach((section) => observer.observe(section));
  }
})();
