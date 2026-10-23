def budget_indices(row, selected_pair, budget, policy="dcrrag"):
    count = len(row["ctxs"])
    if not row.get("bge_ordered"):
        raise ValueError("Budget comparisons require the frozen BGE candidate order")
    if budget < 1 or budget > count:
        raise ValueError("The passage budget must fit the existing candidate pool")
    if (
        len(selected_pair) != 2
        or len(set(selected_pair)) != 2
        or any(index < 0 or index >= count for index in selected_pair)
    ):
        raise ValueError(
            "The frozen main selector must supply two distinct valid passages"
        )
    if policy == "bge":
        return list(range(budget))
    if policy != "dcrrag":
        raise ValueError("Unknown passage-budget policy")
    ranked = sorted(
        range(count),
        key=lambda index: (
            -float(row["ctxs"][index]["student_score"]),
            index,
            str(row["ctxs"][index]["id"]),
        ),
    )
    if budget == 1:
        return ranked[:1]
    selected = list(selected_pair)
    selected.extend(index for index in ranked if index not in selected_pair)
    return sorted(selected[:budget])
