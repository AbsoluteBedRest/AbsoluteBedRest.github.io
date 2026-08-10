---
layout: cv-home
title: Curriculum Vitae
permalink: /cv/
---

{% include widgets/profile_card.html %}

<div class="my-3"></div>

{% include widgets/experience_card.html %}

{% assign selected_publications = site.publications
  | where: "selected", true
  | sort: "date"
  | reverse
%}

{% include widgets/publication_card.html
  publications=selected_publications
%}