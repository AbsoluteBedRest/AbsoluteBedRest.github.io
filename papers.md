---
layout: blog
title: Papers
permalink: /papers/
pagination:
  enabled: true
  category: papers
---

{% assign papers = site.posts
  | where_exp: "post", "post.categories contains 'papers'"
  | sort: "date"
  | reverse %}

<div class="timeline">

{% assign current_year = "" %}

{% for post in papers %}
  {% assign year = post.date | date: "%Y" %}

  {% if year != current_year %}
    <div class="timeline-year">{{ year }}</div>
    {% assign current_year = year %}
  {% endif %}

  <div class="timeline-item">
    <div class="timeline-date">
      {{ post.date | date: "%d %b" }}
    </div>
    <div class="timeline-content">
      <a href="{{ post.url }}">{{ post.title }}</a>
      {% if post.venue %}
        <div class="timeline-venue">{{ post.venue }}</div>
      {% endif %}
    </div>
  </div>

{% endfor %}

</div>
