def measured_entity(tokens, unit_gazetteer, date_gazetteer, time_gazetteer):
        measured_entities = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.replace(',', '').replace('.', '').isdigit():
                measured_entity_parts = [token]
                j = i + 1
                while j < len(tokens):
                    next_token = tokens[j].lower()
                    if next_token in unit_gazetteer or next_token in {"mln", "billion", "thousand", "million"}:
                        measured_entity_parts.append(tokens[j])
                        j += 1 
                    else:
                        break  
                measured_entities.append(' '.join(measured_entity_parts))
                i = j - 1  
            elif tokens[i].lower() in date_gazetteer:
                measured_entities.append(f"{tokens[i]} (date)")
            elif tokens[i].lower() in time_gazetteer:
                measured_entities.append(f"{tokens[i]} (time)")
            else:
                measured_entities.append(token)
            i += 1
        return measured_entities
