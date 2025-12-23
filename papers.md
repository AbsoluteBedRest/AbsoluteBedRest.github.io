---
layout: page
title: Papers
permalink: /papers/
---

{% assign papers = site.posts 
   | where_exp: "post", "post.categories contains 'papers'" 
   | sort: "date" 
   | reverse %}

{% assign current_year = "" %}

{% for post in papers %}
  {% assign year = post.date | date: "%Y" %}

  {% if year != current_year %}
## {{ year }}
  {% assign current_year = year %}
  {% endif %}

<div style="margin-bottom: 1.8rem; padding-bottom: 1.2rem; border-bottom: 1px solid #eee;">

  <div style="font-size: 0.9rem; color: #999;">
    {{ post.date | date: "%Y · %m" }}
    {% if post.venue %}
      · <em>{{ post.venue }}</em>
    {% endif %}
  </div>

  <div style="font-size: 1.15rem; font-weight: 600; margin-top: 0.3rem;">
    <a href="{{ post.url }}">{{ post.title }}</a>
  </div>

  {% if post.excerpt %}
  <div style="color: #666; margin-top: 0.4rem; font-size: 0.95rem;">
    {{ post.excerpt | strip_html | truncate: 160 }}
  </div>
  {% endif %}

</div>
{% endfor %}
