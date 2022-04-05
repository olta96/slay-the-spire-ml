package RecommenderSystem;

import java.util.ArrayList;

public class State {

    private Deck deck;
    private Relics relics;
    private Choices choices;
    private int floor;

    public void setDeck(ArrayList<String> cards) {
        this.deck = new Deck(cards);
    }

    public void setRelics(ArrayList<String> relics) {
        this.relics = new Relics(relics);
    }

    public void setChoices(ArrayList<String> choices) {
        this.choices = new Choices(choices);
    }

    public void setFloor(int floor) {
        this.floor = floor;
    }

    public String getJSON() {
        StringBuilder stringBuilder = new StringBuilder();

        stringBuilder.append("{\"deck\":");
        appendNamesAsJSONList(stringBuilder, deck.getCards());
        stringBuilder.append(",\"relics\":");
        appendNamesAsJSONList(stringBuilder, relics.getRelics());
        stringBuilder.append(",\"available_choices\":");
        appendNamesAsJSONList(stringBuilder, choices.getChoices());
        stringBuilder
                .append(",\"floor\": ")
                .append(this.floor)
                .append("}");

        return stringBuilder.toString();
    }

    private void appendNamesAsJSONList(StringBuilder toAppendTo, ArrayList<String> names) {
        toAppendTo.append("[");
        for (int i = 0; i < names.size(); i++) {
            toAppendTo.append("\"").append(names.get(i)).append("\"");
            if (i != names.size() - 1)
                toAppendTo.append(",");
        }
        toAppendTo.append("]");
    }
}
