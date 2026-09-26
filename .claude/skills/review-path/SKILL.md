---
name: review-path
description: Review a learning path in paths/ for sequencing, prerequisite gaps, difficulty jumps, time budget, and resource-type balance, then propose or apply fixes. Use when the user asks to review, reorder, or sanity-check a path or the whole curriculum.
---

# Review a learning path

1. Read the path file and every lesson it links to, in order.
2. Build a concept dependency list: for each lesson, what it *uses* vs what it *introduces*.
   Flag any concept that's used before it's introduced.
3. Check **difficulty slope**: no lesson should jump more than one level (L1→L3) from its predecessor.
4. Check **resource balance** per lesson: at least one Read (text) step, at least one Build step,
   and video optional. Flag lessons where video is the only way in.
5. Check the **time budget**: sum the estimates and compare with the path's stated duration.
6. Check **redundancy**: the same chapter assigned twice without a reason ("revisit").
7. Output a table: `Lesson | Issue | Severity (blocker/major/minor) | Proposed fix`.
   Then apply the non-controversial fixes. Ask before reordering lessons across paths.
