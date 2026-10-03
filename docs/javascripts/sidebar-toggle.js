/* 왼쪽 목차 접기/펼치기. 상태는 localStorage에 저장해 페이지 이동 후에도 유지한다. */
(function () {
  var KEY = "hwm.sidebarCollapsed"
  var root = document.documentElement
  var buttons = document.querySelectorAll(".sidebar-toggle")

  function update() {
    var collapsed = root.classList.contains("sidebar-collapsed")
    var label = (collapsed ? "목차 펼치기" : "목차 접기") + " ([)"
    buttons.forEach(function (button) {
      button.setAttribute("aria-pressed", String(collapsed))
      button.setAttribute("aria-label", label)
      button.setAttribute("title", label)
    })
  }

  function toggle() {
    var collapsed = root.classList.toggle("sidebar-collapsed")
    try {
      localStorage.setItem(KEY, collapsed ? "1" : "0")
    } catch (e) {}
    update()
  }

  buttons.forEach(function (button) {
    button.addEventListener("click", toggle)
  })

  /* "[" 키로 토글한다. 검색창 등 입력 중일 때와 조합키를 누른 경우는 무시한다. */
  document.addEventListener("keydown", function (event) {
    if (event.key !== "[" || event.ctrlKey || event.metaKey || event.altKey) return
    var target = event.target
    if (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return
    if (!window.matchMedia("(min-width: 76.25em)").matches) return
    event.preventDefault()
    toggle()
  })

  update()
})()
