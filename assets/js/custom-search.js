(function ($) {
  $(function () {

    const $searchContent = $(".search-content");
    const $initialContent = $(".initial-content");
    const $body = $("body");
    const $searchToggle = $(".search__toggle");


    function updateBodyState() {
      const isOpen = $searchContent.hasClass("is--visible");

      $body.toggleClass("search-modal-open", isOpen);
    }


    function closeSearch() {
      $searchContent.removeClass("is--visible");
      $initialContent.removeClass("is--hidden");

      $body.removeClass("search-modal-open");

      $searchToggle.trigger("focus");
    }


    /*
     * 기존 Minimal Mistakes 검색 버튼은 그대로 사용한다.
     * 기본 script가 먼저 is--visible을 toggle한 뒤
     * body 상태만 동기화한다.
     */
    $searchToggle.on("click.customSearch", function () {
      updateBodyState();
    });


    /*
     * X 버튼
     */
    $(document).on(
      "click.customSearch",
      ".custom-search__close",
      function () {
        closeSearch();
      }
    );


    /*
     * 검색 panel 바깥의 어두운 영역 클릭
     */
    $searchContent.on("click.customSearch", function (event) {

      if (event.target === this) {
        closeSearch();
      }

    });


    /*
     * ESC
     *
     * Minimal Mistakes 자체에서도 ESC를 처리하지만
     * body scroll 상태까지 확실하게 복구한다.
     */
    $(document).on("keyup.customSearch", function (event) {

      if (event.key === "Escape") {
        $body.removeClass("search-modal-open");
      }

    });


    /*
     * 검색어가 완전히 비었을 때
     * "0 Result(s) found"를 지운다.
     */
    $("#search").on("keyup.customSearch", function () {

      if (!this.value.trim()) {
        $("#results").empty();
      }

    });

  });
})(jQuery);