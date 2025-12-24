---
layout: page
title: Papers
permalink: /papers/
---

{% assign papers_count = site.categories.papers | size %}

# Papers {% if papers_count > 0 %}({{ papers_count }}){% endif %}

{% assign papers = site.categories.papers | sort: "date" | reverse %}

{% if papers.size == 0 %}
<p>아직 등록된 논문이 없습니다.</p>
{% endif %}

{% for post in papers %}
<article class="post-item">
  <h2>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
  </h2>

  <div class="post-meta">
    📅 {{ post.date | date: "%Y년 %m월 %d일" }}
  </div>

  {% if post.description %}
  <p class="post-description">
    {{ post.description }}
  </p>
  {% endif %}

  {% if post.tags %}
  <div class="post-tags">
    {% for tag in post.tags %}
      <span class="tag">{{ tag }}</span>
    {% endfor %}
  </div>
  {% endif %}
</article>

<hr />
{% endfor %}
